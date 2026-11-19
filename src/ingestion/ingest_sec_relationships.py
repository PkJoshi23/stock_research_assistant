"""SEC EDGAR relationships: officers, subsidiaries and disclosed suppliers.

Why this module exists
----------------------
yfinance gives a price and a sector label. It says nothing about *corporate
structure* - who runs the company, what it owns, and which vendors it says it
depends on. That is exactly the connected-context data this graph is built for,
and it is the only source here that is a primary document rather than a feed.

What it writes
--------------
    (:Person)-[:WORKS_FOR {role, source: 'DEF 14A'}]->(:Stock)
    (:Stock)-[:OWNS]->(:Subsidiary {name})
    (:Stock|Supplier)-[:SUPPLIES_TO {source: '10-K'}]->(:Stock)

Design decisions (each one fixes a failure in the first draft)
--------------------------------------------------------------
* **Bounded text windows, not the document head.** The earlier version sent the
  first 4 000 characters of the filing. For a 10-K that is the cover page and
  the legal boilerplate, so the extractor had nothing to find and returned
  empty lists while looking like it worked. Here the text is sliced *around the
  section heading* (director nominees, subsidiaries, suppliers) and only then
  sent to the model.
* **Fixed relationship types.** The earlier code built the edge type from the
  model's role text - ``MERGE (p)-[:{role}]->(s)`` - producing ``[:CEO]``,
  ``[:BOARD_OF_DIRECTORS]`` and an unbounded schema that no Cypher query can
  rely on. Role is now a *property* on a single ``WORKS_FOR`` edge.
* **Suppliers are resolved against the graph first.** A 10-K naming "TSMC"
  should point at the existing ``Stock:TSM`` node, not create a parallel
  ``Entity:TSMC`` that the agent will never join to. Unresolved names (private
  vendors, "cloud providers") become ``Supplier`` nodes, which is honest about
  what a filing gives us.
* **Idempotent.** Every write is a MERGE, and existing edges are never
  relabelled, so re-running does not churn the graph.

Run:
    python -m src.ingestion.ingest_sec_relationships
    python -m src.ingestion.ingest_sec_relationships --tickers NVDA AAPL
    python -m src.ingestion.ingest_sec_relationships --reset   # drop SEC nodes

Requires SEC_CONTACT_EMAIL in .env: EDGAR rejects requests without a real
contact address, and a placeholder is not worth committing to git.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bs4 import BeautifulSoup

from src import config, llm, relate_db, safety
from src.entities import resolve_tracked_stock
from src.schema import (
    PERSON,
    REL_OWNS,
    REL_SUPPLIES_TO,
    REL_WORKS_FOR,
    STOCK,
    SUBSIDIARY,
    SUPPLIER,
)

#: Where sec-edgar-downloader puts filings, relative to the download folder.
#: Layout: sec-edgar-filings/<ticker>/<form>/<accession>/<document>
_FILINGS_SUBDIR = "sec-edgar-filings"

#: Document names, most useful first.
#:
#: With the default ``download_details=False`` (what we want - one request per
#: filing) version 5.x saves the whole submission as ``full-submission.txt``.
#: ``primary-document.*`` only appears when details are downloaded, but it is
#: kept here so a hand-placed or older cache still works.
_PRIMARY_DOCUMENTS = (
    "full-submission.txt",
    "primary-document.html",
    "primary-document.htm",
)

#: Section headings to search for, most specific first. A window is taken from
#: the first match of the first pattern that matches at all.
PROXY_PATTERNS = (
    r"nominees?\s+for\s+election",
    r"director\s+nominees",
    r"election\s+of\s+directors",
    r"board\s+of\s+directors",
)
ANNUAL_SUBSIDIARY_PATTERNS = (
    r"subsidiaries\s+of\s+the\s+registrant",
    r"list\s+of\s+subsidiaries",
    r"\bsubsidiar(?:y|ies)\b",
)
ANNUAL_SUPPLIER_PATTERNS = (
    r"single[- ]source\s+suppliers?",
    r"depend(?:s|ing)?\s+(?:on|upon)\s+(?:a\s+)?(?:limited\s+number\s+of\s+)?suppliers?",
    r"\bsuppliers?\b",
    r"\bfoundr(?:y|ies)\b",
)

#: Sentences that actually *name* dependencies.
#:
#: The generic ``\bsuppliers?\b`` pattern is a poor anchor on its own: in NVIDIA's
#: 10-K the first match is automotive boilerplate ("tier-1 suppliers"), while the
#: dependency paragraph - the one that names TSMC, Samsung, SK Hynix, Micron and
#: Hon Hai - opens with "we utilize foundries, such as ...". Anchoring on the
#: construction ("utilize/purchase ... such as", "contract manufacturers such as")
#: lands the window on the paragraph with names in it.
ANNUAL_DEPENDENCY_PATTERNS = (
    # Order matters: section_windows fills its budget in pattern order, and the
    # foundry sentence is the one that names TSMC *and* Samsung. Anchoring on
    # contract manufacturers first starts the window after TSMC and loses it.
    r"foundr(?:y|ies),?\s+such\s+as",
    # Allow an adverb inside the verb phrase: AMD writes "We *currently* depend
    # on GLOBALFOUNDRIES", which a rigid "we depend on" pattern never matches.
    r"we\b[^.]{0,40}?\b(?:rely|depend)\s+(?:on|upon)\b",
    r"we\s+(?:purchase|procure|source|obtain)\b[^.]{0,120}?\bfrom\b",
    r"(?:raw materials|components|wafers|memory|packaging)\b[^.]{0,100}?\b(?:from|supplied\s+by)\b",
    r"contract\s+manufacturers?,?\s+such\s+as",
    # 20-F filings phrase it as partnerships rather than suppliers; Nebius says
    # its hardware is "supported by our long-standing partnerships ... such as
    # NVIDIA". Without this, the whole foreign-issuer half of the crawl is empty.
    r"(?:partnerships?|collaborations?|agreements?)\b[^.]{0,160}?such\s+as",
    # Keep this one bounded to a supply-ish noun: the unbounded form anchored on
    # "we depend on our employees" and pushed the supplier paragraph out of the
    # window budget, which is worse than missing it.
    r"depend(?:ence)?\s+(?:on|upon)\b[^.]{0,120}?\b(?:suppliers?|vendors?|partners?|foundr(?:y|ies))",
    r"single[- ]source\s+suppliers?",
)

_EXTRACTION_SYSTEM = """You extract structured facts from US SEC filings.

Absolute rules:
- Use ONLY names that appear verbatim in the provided text.
- Never invent, translate, expand or "correct" a company or person name.
- If the text does not contain the requested information, return an empty list.
- Output valid JSON only, matching the requested shape exactly."""


# -----------------------------------------------------------------------------
# Downloading and reading
# -----------------------------------------------------------------------------


def _downloader():
    """Build the EDGAR downloader, failing with an actionable message."""
    from sec_edgar_downloader import Downloader

    if not config.SEC_CONTACT_EMAIL:
        raise RuntimeError(
            "SEC_CONTACT_EMAIL is not set. EDGAR requires a User-Agent with a "
            "real contact address - set SEC_CONTACT_EMAIL in .env "
            "(see .env.example), then re-run."
        )
    config.SEC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return Downloader(
        config.SEC_COMPANY_NAME,
        config.SEC_CONTACT_EMAIL,
        str(config.SEC_CACHE_DIR),
    )


def filing_path(ticker: str, form: str) -> Path | None:
    """Return the newest cached primary document for ``ticker``/``form``.

    Sorted by mtime rather than accession string: the accession embeds the year
    but not the sequence within it, so a lexical sort is not reliably "newest".
    """
    root = config.SEC_CACHE_DIR / _FILINGS_SUBDIR / ticker / form
    if not root.is_dir():
        return None
    accession_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for accession_dir in accession_dirs:
        for name in _PRIMARY_DOCUMENTS:
            candidate = accession_dir / name
            if candidate.exists():
                return candidate
    return None


def html_to_text(html: str) -> str:
    """Flatten a filing to whitespace-normalised text.

    EDGAR filings are tag soup with inline styles and enormous tables; the
    parser choice matters for a 10-K that can exceed 10 MB. lxml is used when
    available purely for speed, with a fall back that keeps the module working
    on a bare install.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:  # noqa: BLE001 - lxml is optional
        soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))


def fetch_filing_text(ticker: str, form: str) -> str:
    """Download (once, cached) and return the filing's text, or "" on failure."""
    downloader = _downloader()
    for attempt in (1, 2):
        try:
            downloader.get(form, ticker, limit=1)
            break
        except Exception as exc:  # noqa: BLE001 - network / EDGAR errors
            # DNS and edge blips are transient (seen as getaddrinfo failures on
            # data.sec.gov); one retry turns a flaky minute into a slow success.
            if attempt == 2:
                print(f"  ! {ticker} {form}: download failed ({type(exc).__name__}: {exc})")
                return ""
            time.sleep(2)

    path = filing_path(ticker, form)
    if path is None:
        print(f"  ! {ticker} {form}: no primary document in the cache")
        return ""

    try:
        return html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
    except OSError as exc:
        print(f"  ! {ticker} {form}: unreadable ({exc})")
        return ""


# -----------------------------------------------------------------------------
# Section selection
# -----------------------------------------------------------------------------


def section_windows(
    text: str, patterns: tuple[str, ...], size: int | None = None, *, max_windows: int = 2
) -> str:
    """Return up to ``max_windows`` slices of ``text`` around matching headings.

    Taking a window *around the heading* is what makes extraction work: the
    first characters of a filing are the cover page, and the last chained
    "subsidiaries" mention is usually a cross-reference.
    """
    size = size or config.SEC_WINDOW_CHARS
    slices: list[str] = []
    consumed: list[tuple[int, int]] = []

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, match.start() - 200)
            end = start + size
            if any(s <= start < e or s < end <= e for s, e in consumed):
                continue
            consumed.append((start, end))
            slices.append(text[start:end])
            if len(slices) >= max_windows:
                return "\n\n".join(slices)
            break  # one window per pattern is enough
    return "\n\n".join(slices)


# -----------------------------------------------------------------------------
# LLM extraction
# -----------------------------------------------------------------------------


def _clean_names(values, *, limit: int, max_length: int = 120) -> list[str]:
    """Normalise a list of names: strip, drop empties/junk, de-duplicate."""
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        name = re.sub(r"\s+", " ", value).strip(" .,;:-")
        if not name or len(name) > max_length or len(name) < 2:
            continue
        if name.lower() in {existing.lower() for existing in cleaned}:
            continue
        cleaned.append(name)
        if len(cleaned) >= limit:
            break
    return cleaned


def extract_officers(text: str, ticker: str) -> list[dict]:
    """Ask the model for officers/directors named in a proxy-statement window."""
    if not text:
        return []
    prompt = (
        f"From this {ticker} proxy statement (DEF 14A), list the directors and "
        "named executive officers.\n\n"
        "Return JSON: {\"executives\": [{\"name\": \"Full Name\", \"role\": \"CEO\"}]}\n"
        "- role must be a short standard title (\"CEO\", \"CFO\", \"Director\", \"Board Member\").\n"
        "- at most 10 people; only those whose full name appears in the text.\n\n"
        + safety.wrap_untrusted(text, "FILING TEXT")
    )
    try:
        data = llm.chat_json(
            prompt, system=f"{_EXTRACTION_SYSTEM}\n\n{safety.UNTRUSTED_DATA_RULE}"
        )
    except (ValueError, RuntimeError) as exc:
        print(f"  ! {ticker} proxy extraction failed ({type(exc).__name__})")
        return []

    officers: list[dict] = []
    for item in data.get("executives", []) or []:
        if not isinstance(item, dict):
            continue
        names = _clean_names([item.get("name")], limit=1)
        if not names:
            continue
        role = re.sub(r"\s+", " ", str(item.get("role") or "Officer")).strip()
        officers.append({"name": names[0], "role": role[:60] or "Officer"})
        if len(officers) >= 10:
            break
    return officers


def extract_dependencies(text: str, ticker: str) -> list[str]:
    """Named companies this filing says it depends on.

    A different question from :func:`extract_structure`: the crawler wants the
    dependency list, not the subsidiary list, and foreign issuers call these
    "partnerships" rather than "suppliers". Asking one prompt for both made the
    model drop NVIDIA in Nebius's 20-F, where the sentence is "our hardware is
    supported by our long-standing partnerships ... such as NVIDIA".
    """
    if not text:
        return []
    prompt = (
        f"From this {ticker} annual report, list the companies it says it depends "
        "on for products, components, manufacturing, cloud capacity or other "
        "critical inputs - suppliers, foundries, contract manufacturers and key "
        "hardware partners.\n\n"
        "Return JSON: {\"suppliers\": [\"Company Name\"]}\n"
        "- only companies named in the text; never infer or generalise.\n"
        "- exclude categories (\"various suppliers\", \"cloud providers\") and "
        "exclude the filer itself.\n"
        "- at most 8 names; [] when the text supports none.\n\n"
        + safety.wrap_untrusted(text, "FILING TEXT")
    )
    try:
        data = llm.chat_json(
            prompt, system=f"{_EXTRACTION_SYSTEM}\n\n{safety.UNTRUSTED_DATA_RULE}"
        )
    except (ValueError, RuntimeError) as exc:
        print(f"  ! {ticker} dependency extraction failed ({type(exc).__name__})")
        return []
    return _clean_names(data.get("suppliers"), limit=8)


def extract_structure(text: str, ticker: str) -> dict[str, list[str]]:
    """Ask the model for subsidiaries and disclosed suppliers from 10-K text."""
    if not text:
        return {"subsidiaries": [], "suppliers": []}
    prompt = (
        f"From this {ticker} 10-K, extract two lists.\n\n"
        "1. subsidiaries: legal entities the company says it owns or that are "
        "listed as its subsidiaries.\n"
        "2. suppliers: specific companies the filing says supply the company with "
        "goods or services (chips, components, cloud capacity, contract "
        "manufacturing). Only named companies. Skip generic phrases like "
        "\"various suppliers\", \"third-party providers\" or industry categories.\n\n"
        "Return JSON: {\"subsidiaries\": [\"Name\"], \"suppliers\": [\"Name\"]}\n"
        "- at most 8 subsidiaries and 5 suppliers; [] when the text supports none.\n\n"
        + safety.wrap_untrusted(text, "FILING TEXT")
    )
    try:
        data = llm.chat_json(
            prompt, system=f"{_EXTRACTION_SYSTEM}\n\n{safety.UNTRUSTED_DATA_RULE}"
        )
    except (ValueError, RuntimeError) as exc:
        print(f"  ! {ticker} 10-K extraction failed ({type(exc).__name__})")
        return {"subsidiaries": [], "suppliers": []}

    return {
        "subsidiaries": _clean_names(data.get("subsidiaries"), limit=8),
        "suppliers": _clean_names(data.get("suppliers"), limit=5),
    }


# -----------------------------------------------------------------------------
# Graph writes
# -----------------------------------------------------------------------------


# ``resolve_tracked_stock`` is imported above from ``src.entities`` and
# re-exported here on purpose: other loaders need the same alias matcher,
# and one implementation is the only way the two paths cannot disagree. The
# original definition lived in this module; it is imported (not redefined) so
# ``sec.resolve_tracked_stock`` keeps working for existing callers and tests.


def _name_parts(name: str) -> list[str]:
    """Lower-case, strip punctuation, split into words."""
    cleaned = re.sub(r"[^a-z\s]", "", name.lower())
    return [part for part in cleaned.split() if part]


def same_person(first: str, second: str) -> bool:
    """Heuristic identity check for the same human written two ways.

    EDGAR says "Jen-Hsun Huang"; the curated graph says "Jensen Huang". A strict
    name match would create a second Person node for the same CEO, which looks
    broken in the visualisation and breaks "who runs X" queries.

    The rule is deliberately narrow: identical surname *and* first names that
    share their first three letters. Two colleagues called "John Smith" and
    "James Smith" stay separate, which is the failure that matters.
    """
    left, right = _name_parts(first), _name_parts(second)
    if len(left) < 2 or len(right) < 2:
        return False
    if left[-1] != right[-1]:
        return False
    return left[0][:3] == right[0][:3]


def resolve_person(conn, ticker: str, name: str) -> str:
    """Return the existing name for this person at this company, if any.

    Scoped to people already linked to ``ticker``: the same surname elsewhere in
    the watchlist is very unlikely to be the same person, and a global fuzzy
    match would quietly merge unrelated directors.
    """
    rows = relate_db.run(
        conn,
        f"""
        MATCH (p:{PERSON})-[:{REL_WORKS_FOR}]->(s:{STOCK} {{ticker: $ticker}})
        RETURN p.name AS name
        """,
        {"ticker": ticker},
    )
    for row in rows:
        existing = row["name"] or ""
        if existing == name or same_person(existing, name):
            return existing
    return name


def write_officer(conn, ticker: str, name: str, role: str) -> str:
    """Link a person to a stock, preserving an existing curated role.

    Returns the node name actually used, which may differ from ``name`` when the
    filing spells the person differently from the curated graph.
    """
    name = resolve_person(conn, ticker, name)
    existing = relate_db.run(
        conn, f"MATCH (p:{PERSON} {{name: $name}}) RETURN p.role AS role", {"name": name}
    )
    relate_db.run(conn, f"MERGE (p:{PERSON} {{name: $name}})", {"name": name})
    if not existing or not existing[0]["role"]:
        relate_db.run(
            conn,
            f"MATCH (p:{PERSON} {{name: $name}}) SET p.role = $role",
            {"name": name, "role": role},
        )
    relate_db.run(
        conn,
        f"""
        MATCH (p:{PERSON} {{name: $name}}), (s:{STOCK} {{ticker: $ticker}})
        MERGE (p)-[r:{REL_WORKS_FOR}]->(s)
        SET r.source = 'DEF 14A'
        """,
        {"name": name, "ticker": ticker},
    )
    return name


def write_subsidiary(conn, ticker: str, name: str) -> None:
    relate_db.run(conn, f"MERGE (sub:{SUBSIDIARY} {{name: $name}})", {"name": name})
    relate_db.run(
        conn,
        f"""
        MATCH (s:{STOCK} {{ticker: $ticker}}), (sub:{SUBSIDIARY} {{name: $name}})
        MERGE (s)-[:{REL_OWNS}]->(sub)
        """,
        {"ticker": ticker, "name": name},
    )


def write_supplier(conn, ticker: str, name: str) -> str:
    """Link a filing-named supplier. Returns a short description for logging."""
    resolved = resolve_tracked_stock(conn, name)

    if resolved and resolved != ticker:
        # The supplier is already in the graph: reuse the stock node so the
        # agent's existing SUPPLIES_TO queries see this relationship too.
        already = relate_db.run(
            conn,
            f"""
            MATCH (sup:{STOCK} {{ticker: $sup}})-[r:{REL_SUPPLIES_TO}]->(:{STOCK} {{ticker: $t}})
            RETURN r.source AS source
            """,
            {"sup": resolved, "t": ticker},
        )
        if already:
            # The edge may be curated. Annotate it as filing-confirmed rather
            # than overwriting `source`, which would misattribute where the
            # relationship originally came from.
            relate_db.run(
                conn,
                f"""
                MATCH (sup:{STOCK} {{ticker: $sup}})-[r:{REL_SUPPLIES_TO}]->(:{STOCK} {{ticker: $t}})
                SET r.critical = true, r.confirmedBy = '10-K'
                """,
                {"sup": resolved, "t": ticker},
            )
        else:
            relate_db.run(
                conn,
                f"""
                MATCH (sup:{STOCK} {{ticker: $sup}}), (s:{STOCK} {{ticker: $t}})
                MERGE (sup)-[r:{REL_SUPPLIES_TO}]->(s)
                SET r.source = '10-K', r.critical = true
                """,
                {"sup": resolved, "t": ticker},
            )
        return f"{resolved} (tracked stock)"

    relate_db.run(conn, f"MERGE (v:{SUPPLIER} {{name: $name}})", {"name": name})
    relate_db.run(
        conn,
        f"""
        MATCH (v:{SUPPLIER} {{name: $name}}), (s:{STOCK} {{ticker: $ticker}})
        MERGE (v)-[r:{REL_SUPPLIES_TO}]->(s)
        SET r.source = '10-K'
        """,
        {"name": name, "ticker": ticker},
    )
    return f"{name} (unlisted supplier)"


def reset_sec_nodes(conn) -> None:
    """Delete SEC-derived nodes. Stocks and curated edges are untouched."""
    relate_db.run(conn, f"MATCH (n:{SUBSIDIARY}) DETACH DELETE n")
    relate_db.run(conn, f"MATCH (n:{SUPPLIER}) DETACH DELETE n")


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def ingest_ticker(conn, ticker: str) -> dict[str, int]:
    """Process one ticker. Returns counts for the run summary."""
    counts = {"officers": 0, "subsidiaries": 0, "suppliers": 0}

    if not relate_db.run(
        conn, f"MATCH (s:{STOCK} {{ticker: $t}}) RETURN s.ticker AS t", {"t": ticker}
    ):
        print(f"  ! {ticker}: no Stock node - run ingest_stock_data first")
        return counts

    print("  proxy statement (DEF 14A): officers and directors")
    proxy_text = fetch_filing_text(ticker, config.SEC_PROXY_FORM)
    proxy_window = section_windows(proxy_text, PROXY_PATTERNS)
    if not proxy_window:
        print("  ~ no director section found; skipping officers")
    else:
        for officer in extract_officers(proxy_window, ticker):
            linked_name = write_officer(conn, ticker, officer["name"], officer["role"])
            counts["officers"] += 1
            print(f"    + {linked_name} -[:WORKS_FOR {{role: '{officer['role']}'}}]-> {ticker}")

    time.sleep(0.5)  # EDGAR allows ~10 req/s; stay well under it.

    print("  annual report (10-K): subsidiaries and suppliers")
    annual_text = fetch_filing_text(ticker, config.SEC_ANNUAL_FORM)
    annual_window = "\n\n".join(
        part
        for part in (
            section_windows(annual_text, ANNUAL_SUBSIDIARY_PATTERNS, max_windows=1),
            section_windows(annual_text, ANNUAL_SUPPLIER_PATTERNS, max_windows=1),
        )
        if part
    )
    if not annual_window:
        print("  ~ no subsidiaries/suppliers section found; skipping")
        return counts

    structure = extract_structure(annual_window, ticker)
    for name in structure["subsidiaries"]:
        write_subsidiary(conn, ticker, name)
        counts["subsidiaries"] += 1
        print(f"    + {ticker} -[:OWNS]-> {name}")
    for name in structure["suppliers"]:
        description = write_supplier(conn, ticker, name)
        counts["suppliers"] += 1
        print(f"    + {ticker} supplier: {description}")

    return counts


def main(tickers: list[str] | None = None, reset: bool = False) -> int:
    targets = tickers or config.WATCHLIST
    conn = relate_db.connect()

    try:
        if reset:
            reset_sec_nodes(conn)
            print("Removed existing Subsidiary/Supplier nodes (stocks kept).")

        print(f"Reading SEC filings for {len(targets)} ticker(s): {', '.join(targets)}")
        totals = {"officers": 0, "subsidiaries": 0, "suppliers": 0}
        for ticker in targets:
            print(f"\n{ticker}")
            try:
                counts = ingest_ticker(conn, ticker)
            except RuntimeError:
                raise  # missing contact email: not recoverable per ticker
            except Exception as exc:  # noqa: BLE001 - one bad filing must not stop the run
                print(f"  ! {ticker}: aborted ({type(exc).__name__}: {exc})")
                continue
            for key, value in counts.items():
                totals[key] += value

        print("\nGraph now contains:")
        for label, count in sorted(relate_db.counts(conn).items()):
            print(f"  {label:<12} {count}")
        print(
            f"\nSEC additions: {totals['officers']} officer links, "
            f"{totals['subsidiaries']} subsidiaries, {totals['suppliers']} suppliers"
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="*", help="subset of the watchlist to process")
    parser.add_argument(
        "--reset", action="store_true", help="delete Subsidiary/Supplier nodes first"
    )
    args = parser.parse_args()
    raise SystemExit(main(tickers=args.tickers, reset=args.reset))
