"""Fast on-demand SEC ingestion for a company the graph does not track yet.

The deep crawler (``discover_supply_chain``) reads a company's *full submission*
- XBRL, exhibits and all, 10-40 MB - converts the whole thing to text and slices
three large windows for the model. That is the right trade for a seed universe
crawled once; it is the wrong trade for "who supplies AAOI?" answered in 4 s.

This module exploits three facts about 10-K filings:

1. the document is **standardised**: Item 1 (Business) names suppliers, Item 1A
   is Risk Factors, Exhibit 21 lists subsidiaries;
2. the **primary document** is 1-3 MB, while the full submission is 10x larger
   and mostly XBRL;
3. therefore we never need the whole file: locate the Item headings, slice
   ~20 k characters, and send only that.

Order of work, cheapest first:

    ticker -> CIK        company_tickers.json (cached, no request)
    CIK -> filing        data.sec.gov/submissions (~1 KB of JSON)
    filing -> bytes      primary document only (1-3 MB)
    bytes -> text        regex de-tag (no DOM, no lxml)
    text -> section      anchor on "Item 1. Business" ... "Item 1A"
    section -> names     deterministic list patterns first
    names -> model        only if determinism found nothing (~1 k tokens)

Subsidiaries come from Exhibit 21 - a plain list - so that costs zero tokens.
"""

from __future__ import annotations

import html
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests

from src import config, entities, llm, relate_db, ticker_directory
from src.ingestion import ingest_sec_relationships as sec
from src.ingestion import ingest_stock_data as stocks
from src.schema import STOCK, SUBSIDIARY, SUPPLIER

#: Hard cap on a downloaded primary document. The largest 10-Ks observed here are
#: ~25 MB (ASML's 20-F) but a 10-K primary document is normally 1-3 MB; anything
#: beyond this is truncated rather than refused, and the report says so.
MAX_DOCUMENT_BYTES = 12 * 1024 * 1024

#: Characters kept from Item 1. Enough for product/supplier description, small
#: enough that the optional model fallback stays around 1 k tokens.
MAX_SECTION_CHARS = 20000

#: Sentinel returned when a section anchor cannot be found.
_NO_SECTION = ""


@dataclass
class FastReport:
    """What the fast path learned, and what it cost."""

    ticker: str
    cik: str | None = None
    company_name: str | None = None
    form: str | None = None
    accession: str | None = None
    filed: str | None = None
    document_bytes: int = 0
    document_truncated: bool = False
    text_chars: int = 0
    section_chars: int = 0
    suppliers: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    subsidiaries: list[str] = field(default_factory=list)
    #: Quoted filing sentence when no supplier names are disclosed - "we depend
    #: on a limited number of suppliers" - so the answer can say *why* it is
    #: empty instead of implying the company has no dependencies.
    evidence: str | None = None
    #: Human-readable reason the fetch failed (HTTP status, missing contact...).
    #: A network failure is a result, not an exception: the UI must render it.
    error: str | None = None
    cached: bool = False
    tokens: int = 0
    elapsed_ms: int = 0
    wrote: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "cik": self.cik,
            "company_name": self.company_name,
            "form": self.form,
            "accession": self.accession,
            "filed": self.filed,
            "document_bytes": self.document_bytes,
            "document_truncated": self.document_truncated,
            "text_chars": self.text_chars,
            "section_chars": self.section_chars,
            "suppliers": self.suppliers,
            "unresolved": self.unresolved,
            "subsidiaries": self.subsidiaries,
            "evidence": self.evidence,
            "error": self.error,
            "cached": self.cached,
            "tokens": self.tokens,
            "elapsed_ms": self.elapsed_ms,
            "wrote": self.wrote,
            "notes": self.notes,
        }


# -----------------------------------------------------------------------------
# HTTP
# -----------------------------------------------------------------------------


def _headers() -> dict[str, str]:
    """EDGAR requires a descriptive User-Agent with a contact address.

    Without one the submissions and Archives endpoints answer 403 - which is
    exactly what the deployed app hit when SEC_CONTACT_EMAIL was unset. The
    marker below makes that state visible in the request rather than pretending
    to be a contact that does not exist.
    """
    contact = config.SEC_CONTACT_EMAIL or "contact-not-configured"
    return {
        "User-Agent": f"{config.SEC_COMPANY_NAME}/1.0 ({contact})",
        "Accept-Encoding": "gzip, deflate",
    }


def _http_error_message(exc: Exception) -> str:
    """Turn a request failure into something a user can act on."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 403:
        return (
            "SEC rejected the request (HTTP 403). EDGAR requires a contact "
            "address: set SEC_CONTACT_EMAIL in the app secrets (e.g. "
            "you@example.com) and reboot."
        )
    if status == 404:
        return "SEC has no document at that address (HTTP 404)."
    detail = str(exc).splitlines()[0][:160]
    return f"SEC request failed ({status or type(exc).__name__}): {detail}"


def get_json(url: str, *, timeout: float | None = None) -> dict:
    response = requests.get(url, headers=_headers(), timeout=timeout or config.HTTP_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def get_text(url: str, *, max_bytes: int = MAX_DOCUMENT_BYTES) -> tuple[str, int, bool]:
    """Fetch a document, returning ``(text, bytes_read, truncated)``.

    Streaming with a byte cap: the point of this module is not to read 40 MB.
    """
    with requests.get(
        url, headers=_headers(), timeout=config.HTTP_TIMEOUT_S, stream=True
    ) as response:
        response.raise_for_status()
        chunks: list[bytes] = []
        total = 0
        truncated = False
        for chunk in response.iter_content(chunk_size=1 << 16):
            if not chunk:
                continue
            remaining = max_bytes - total
            if len(chunk) >= remaining:
                chunks.append(chunk[:remaining])
                total += remaining
                truncated = True
                break
            chunks.append(chunk)
            total += len(chunk)
    encoding = response.encoding or "utf-8"
    try:
        text = b"".join(chunks).decode(encoding, errors="replace")
    except LookupError:
        text = b"".join(chunks).decode("utf-8", errors="replace")
    return text, total, truncated


# -----------------------------------------------------------------------------
# Filing discovery
# -----------------------------------------------------------------------------


def resolve_cik(ticker: str) -> tuple[str, str] | None:
    """Map a ticker to ``(10-digit CIK, company title)`` using SEC's own map."""
    wanted = ticker.strip().upper()
    for entry in ticker_directory.load_directory():
        if str(entry.get("ticker", "")).upper() == wanted:
            cik = str(entry.get("cik") or "").zfill(10)
            return (cik, entry.get("title") or "")
    return None


@dataclass(frozen=True)
class Filing:
    form: str
    accession: str
    primary_document: str
    filed: str
    cik: str

    @property
    def document_url(self) -> str:
        cik = str(int(self.cik)) if self.cik.isdigit() else self.cik
        accession = self.accession.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
            f"{self.primary_document}"
        )

    @property
    def index_url(self) -> str:
        cik = str(int(self.cik)) if self.cik.isdigit() else self.cik
        accession = self.accession.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json"


def latest_filing(cik: str, forms: tuple[str, ...] = ("10-K", "20-F")) -> Filing | None:
    """Return the most recent filing of any of ``forms`` for a CIK.

    One small JSON request; the submissions endpoint carries the primary document
    name, so no HTML index page has to be scraped.
    """
    payload = get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    recent = (payload.get("filings") or {}).get("recent") or {}
    wanted = {form.upper() for form in forms}

    matches: list[Filing] = []
    for form, accession, document, filed in zip(
        recent.get("form") or [],
        recent.get("accessionNumber") or [],
        recent.get("primaryDocument") or [],
        recent.get("filingDate") or [],
    ):
        if str(form).upper() in wanted and document:
            matches.append(
                Filing(
                    form=str(form).upper(),
                    accession=str(accession),
                    primary_document=str(document),
                    filed=str(filed),
                    cik=cik,
                )
            )
    if not matches:
        return None
    # SEC returns recent filings newest-first, but that is a presentation
    # contract; selecting by date makes the function independent of it.
    return max(matches, key=lambda filing: filing.filed)


# -----------------------------------------------------------------------------
# Text handling
# -----------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_WS = re.compile(r"\s+")


def de_tag(raw: str) -> str:
    """Strip HTML to text with regexes - no DOM, no lxml, no 4 M-char tree."""
    text = _SCRIPT.sub(" ", raw)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    return _WS.sub(" ", text).strip()


_ITEM1 = re.compile(r"item\s*1\.?\s*business\b", re.IGNORECASE)
_ITEM1A = re.compile(r"item\s*1a\.?\s*risk\s*factors", re.IGNORECASE)


def business_section(text: str, *, size: int = MAX_SECTION_CHARS) -> str:
    """Slice Item 1 (Business) out of a de-tagged filing.

    The first ``Item 1. Business`` hit is usually the table of contents, where
    ``Item 1A`` follows within a few dozen characters. The body section is the
    first occurrence whose next ``Item 1A`` is comfortably far away; that single
    heuristic removes the need to parse the TOC.
    """
    starts = [match.start() for match in _ITEM1.finditer(text)]
    if not starts:
        return _NO_SECTION

    for start in starts:
        risk = _ITEM1A.search(text, start)
        if risk and risk.start() - start > 500:
            return text[start : min(risk.start(), start + size)]

    last = starts[-1]
    return text[last : last + size]


# -----------------------------------------------------------------------------
# Deterministic extraction
# -----------------------------------------------------------------------------

_LIST_LEAD = re.compile(
    r"(?:foundr(?:y|ies)|suppliers?|contract\s+manufacturers?|vendors?|"
    r"component\s+suppliers?|assembly\s+and\s+test\s+subcontractors)"
    r"[^.]{0,160}?(?:such\s+as|include|including|from)\s+([^.]{5,400})\.",
    re.IGNORECASE,
)
_PURCHASE_FROM = re.compile(
    r"(?:purchase|procure|source|obtain)s?\s+[^.]{0,100}?\bfrom\s+([^.]{5,400})\.",
    re.IGNORECASE,
)

_STOP_NAME = {
    "our", "the", "a", "an", "and", "or", "other", "third", "parties", "party",
    "several", "various", "time", "them", "us", "we", "its", "their",
}

#: A real name starts like a proper noun; a category phrase starts with
#: "suppliers/vendors/..." and usually carries a relative clause
#: ("suppliers who have domestic manufacturing capacity").
_STARTS_GENERIC = re.compile(
    r"^(?:the\s+)?(?:suppliers?|vendors?|manufacturers?|foundr(?:y|ies)|"
    r"subcontractors?|customers?|companies|providers?|parties|sources?|partners?)"
    r"\b",
    re.IGNORECASE,
)
_RELATIVE_CLAUSE = re.compile(r"\b(?:who|whom|whose|which|that)\b", re.IGNORECASE)

#: A name whose last word is a category is a category, not a company:
#: "AI model makers", "NVIDIA as direct customers", "ODMs", "CSPs".
#: Matching the last word (not any word) keeps real names such as
#: "Taiwan Semiconductor Manufacturing Company Limited".
_GENERIC_LAST_WORD = re.compile(
    r"\b(?:odms?|oems?|csps?|isvs?|vars?|sdk|makers?|vendors?|suppliers?|"
    r"providers?|manufacturers?|companies|company|customers?|players?|firms?|"
    r"operators?|builders?|partners?|integrators?|sellers?|resellers?|"
    r"distributors?|suppliers?)$",
    re.IGNORECASE,
)
_PHRASE_AS = re.compile(r"\bas\b", re.IGNORECASE)

#: Sentences that prove the filing *discusses* suppliers without naming them.
#: AAOI is the canonical case: "a sole source or ... qualified only from a single
#: supplier", "we depend on a limited number of suppliers" - no names anywhere.
#:
#: Deliberately keyword-first: an earlier version matched
#: ``[^.]*\\b(keyword)\\b[^.]*\\.`` and backtracked for 26 seconds on a 393 k-char
#: filing. Finding the keyword, then expanding to sentence boundaries, is linear.
_EVIDENCE_KEYWORD = re.compile(
    r"(?:sole[-\s]?source|single[-\s]?source|limited number of suppliers|"
    r"depend(?:s|ing)? on (?:a )?(?:limited number of )?suppliers)",
    re.IGNORECASE,
)
_PAGE_NOISE = re.compile(r"^(?:\d{1,3}\s+)?(?:table of contents\s+)?", re.IGNORECASE)


def supplier_evidence(text: str) -> str | None:
    """First sentence that discloses a dependency without naming the vendor."""
    match = _EVIDENCE_KEYWORD.search(text)
    if not match:
        return None
    start = text.rfind(".", 0, match.start()) + 1
    end = text.find(".", match.end())
    end = end + 1 if end != -1 else len(text)
    sentence = re.sub(r"\s+", " ", text[start:end]).strip()
    sentence = _PAGE_NOISE.sub("", sentence).strip()
    return sentence[:300] or None

#: ``", Inc."`` is part of a name, not a list separator. Without this, the split
#: turns "Samsung Electronics Co., Ltd." into "Samsung Electronics Co." + "Ltd".
_LEGAL_SUFFIX = re.compile(
    r",\s*(inc|incorporated|ltd|limited|llc|corp|corporation|co|company|plc|"
    r"n\.v\.|b\.v\.|s\.a\.|ag|se|l\.p\.)\b\.?",
    re.IGNORECASE,
)

#: Text after a list that is prose, not an entity: "..., to produce our wafers".
_TRAILING_CLAUSE = re.compile(r"\s+(?:to|for|which|that|in\s+order|as\s+well\s+as)\s+", re.IGNORECASE)


def _clean_names(chunk: str) -> list[str]:
    """Turn ``"TSMC, Samsung Electronics and Hon Hai"`` into three names."""
    chunk = re.sub(r"\b(?:respectively|among others|etc)\b", "", chunk, flags=re.IGNORECASE)
    # Cut prose tails *before* suffix repair, so "Ltd., to produce" stays a name.
    chunk = _TRAILING_CLAUSE.split(chunk, maxsplit=1)[0]
    chunk = _LEGAL_SUFFIX.sub(lambda match: " " + match.group(1), chunk)
    parts = re.split(r",\s*|\s+and\s+|\s*;\s*", chunk)
    names: list[str] = []
    for part in parts:
        name = part.strip(" .;:()")
        name = re.sub(r"\s+", " ", name)
        if not (2 <= len(name) <= 80):
            continue
        if name.lower() in _STOP_NAME:
            continue
        if _STARTS_GENERIC.match(name) or _RELATIVE_CLAUSE.search(name):
            continue
        if _GENERIC_LAST_WORD.search(name) or _PHRASE_AS.search(name):
            continue
        if not (name[0].isupper() or name[0].isdigit()):
            continue
        if sum(character.isalpha() for character in name) < 3:
            continue
        names.append(name)
    # De-duplicate, order preserved.
    return list(dict.fromkeys(names))


#: Periods that belong to a legal suffix ("Co.", "Ltd.", "Inc.") otherwise end
#: the sentence scan early and truncate the name to "Samsung Electronics Co".
_LEGAL_PERIOD = re.compile(
    r"\b(Inc|Ltd|Co|LLC|Corp|Corporation|plc|N\.V|B\.V|S\.A)\.", re.IGNORECASE
)


def deterministic_suppliers(section: str) -> list[str]:
    """Pull supplier names out of list-shaped sentences. No tokens spent."""
    if not section:
        return []

    text = _LEGAL_PERIOD.sub(r"\1", section)
    names: list[str] = []
    for pattern in (_LIST_LEAD, _PURCHASE_FROM):
        for match in pattern.finditer(text):
            names.extend(_clean_names(match.group(1)))
    return list(dict.fromkeys(names))


# -----------------------------------------------------------------------------
# Exhibit 21 (subsidiaries) - a plain list, so zero tokens
# -----------------------------------------------------------------------------

_BLOCK_TAG = re.compile(r"</?(?:p|div|li|tr|br|h[1-6])\b[^>]*>", re.IGNORECASE)


def parse_exhibit21(raw: str) -> list[str]:
    """Parse a subsidiaries exhibit into names."""
    text = _BLOCK_TAG.sub("\n", raw)
    text = _SCRIPT.sub(" ", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)

    names: list[str] = []
    for line in text.split("\n"):
        candidate = _WS.sub(" ", line).strip(" .;:—–-\t")
        # Exhibit 21 rows usually end with the jurisdiction: "Name - Country".
        candidate = re.split(r"\s+[-\u2013\u2014]\s+", candidate)[0].strip(" .;:")
        if not (3 <= len(candidate) <= 120):
            continue
        if not re.search(r"[A-Za-z]{3}", candidate):
            continue
        if candidate.lower() in {"name", "jurisdiction", "subsidiaries of the registrant"}:
            continue
        if candidate.lower().startswith(("exhibit 21", "subsidiaries of")):
            continue
        names.append(candidate)
    return list(dict.fromkeys(names))


def exhibit21_names(filing: Filing) -> list[str]:
    """Fetch and parse the Exhibit 21 document, if the filing has one."""
    try:
        index = get_json(filing.index_url)
    except Exception:  # noqa: BLE001 - exhibit is best effort
        return []

    for item in (index.get("directory") or {}).get("item") or []:
        kind = str(item.get("type") or "").upper()
        name = str(item.get("name") or "")
        lowered = name.lower()
        if kind.startswith("EX-21") or "ex21" in lowered or "exhibit21" in lowered:
            base = filing.index_url.rsplit("/", 1)[0]
            try:
                raw, _, _ = get_text(f"{base}/{name}", max_bytes=2 * 1024 * 1024)
            except Exception:  # noqa: BLE001
                return []
            return parse_exhibit21(raw)
    return []


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

#: De-tagged text cache. A second question about the same company should cost no
#: download at all; the key is the accession, so a new annual report invalidates.
def _cache_path(filing: Filing) -> Path:
    return config.PROJECT_ROOT / ".cache" / "fast-sec" / f"{filing.cik}-{filing.accession}.txt"


def _load_cached_text(filing: Filing) -> str:
    path = _cache_path(filing)
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _store_cached_text(filing: Filing, text: str) -> None:
    try:
        path = _cache_path(filing)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        pass  # cache is an optimisation, never a failure


def _resolve_name(conn, name: str, directory: ticker_directory.TickerDirectory) -> str | None:
    tracked = entities.resolve_tracked_stock(conn, name)
    if tracked:
        return tracked
    return directory.lookup(name)


def ingest(
    conn,
    ticker: str,
    *,
    write: bool = True,
    allow_model: bool = True,
) -> FastReport:
    """Run the fast path for one ticker and (optionally) write it to the graph."""
    started = time.perf_counter()
    report = FastReport(ticker=ticker.strip().upper())

    resolved = resolve_cik(report.ticker)
    if not resolved:
        report.notes.append("ticker not found in SEC's registrant map")
        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return report
    report.cik, report.company_name = resolved

    try:
        filing = latest_filing(report.cik)
    except requests.RequestException as exc:
        report.error = _http_error_message(exc)
        report.notes.append(report.error)
        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return report
    if not filing:
        report.notes.append("no 10-K or 20-F in the recent filings list")
        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return report
    report.form, report.accession, report.filed = filing.form, filing.accession, filing.filed

    text = _load_cached_text(filing)
    if text:
        report.cached = True
    else:
        try:
            raw, size, truncated = get_text(filing.document_url)
        except requests.RequestException as exc:
            # A 403 here is the expected outcome when SEC_CONTACT_EMAIL is unset.
            # Report it; never let a network failure reach the UI as a traceback.
            report.error = _http_error_message(exc)
            report.notes.append(report.error)
            report.elapsed_ms = int((time.perf_counter() - started) * 1000)
            return report
        report.document_bytes, report.document_truncated = size, truncated
        if truncated:
            report.notes.append(f"document truncated at {size} bytes")
        text = de_tag(raw)
        _store_cached_text(filing, text)

    report.text_chars = len(text)
    section = business_section(text)
    report.section_chars = len(section)

    suppliers = deterministic_suppliers(section)
    if not suppliers:
        # Free, still no tokens: suppliers are sometimes named outside Item 1
        # (Risk Factors, MD&A), and the text is already in memory.
        suppliers = deterministic_suppliers(text)

    if not suppliers:
        report.evidence = supplier_evidence(text)
        window = section
        if not window and report.evidence:
            anchor = text.find(report.evidence[:60])
            window = text[max(0, anchor - 500) : anchor + 7000] if anchor >= 0 else text[:8000]
        if window and allow_model:
            llm.start_trace()
            try:
                # Smaller slice for the model: the deterministic pass found
                # nothing, so pay for prose understanding on a bounded window.
                suppliers = sec.extract_dependencies(window[:8000], report.ticker)
            finally:
                usage = llm.finish_trace()  # thread-local: no handle to pass
            report.tokens = int(usage.get("total_tokens", 0))
    report.suppliers = suppliers

    if filing.form == "10-K":
        report.subsidiaries = exhibit21_names(filing)
    else:
        report.notes.append(
            f"{filing.form} has no Exhibit 21; subsidiaries not extracted"
        )

    if not suppliers:
        report.notes.append(
            "the filing names no suppliers"
            + ("; evidence captured" if report.evidence else "")
        )
    if not section:
        report.notes.append("Item 1. Business anchor not found")

    if not write:
        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return report

    # --- write ---------------------------------------------------------------
    try:
        snapshot = stocks.fetch_snapshot(report.ticker)
    except Exception as exc:  # noqa: BLE001 - yfinance can fail in many ways
        report.notes.append(f"snapshot unavailable ({type(exc).__name__}); wrote ticker only")
        snapshot = None
    if snapshot:
        stocks.upsert_stock(conn, snapshot)
    else:
        relate_db.query(
            conn,
            f"MERGE (s:{STOCK} {{ticker: $ticker}}) SET s.name = coalesce(s.name, $name)",
            {"ticker": report.ticker, "name": report.company_name},
        )
        report.notes.append("snapshot unavailable; wrote ticker and name only")

    directory = ticker_directory.TickerDirectory.load()
    for name in suppliers:
        supplier_ticker = _resolve_name(conn, name, directory)
        if supplier_ticker:
            relate_db.query(
                conn,
                f"""
                MERGE (sup:{STOCK} {{ticker: $ticker}})
                ON CREATE SET sup.name = $name
                WITH sup
                MATCH (cust:{STOCK} {{ticker: $customer}})
                MERGE (sup)-[r:SUPPLIES_TO]->(cust)
                SET r.discovered = true,
                    r.source = coalesce(r.source, $form),
                    r.accession = $accession,
                    r.filed = $filed
                """,
                {
                    "ticker": supplier_ticker,
                    "name": name,
                    "customer": report.ticker,
                    "form": filing.form,
                    "accession": filing.accession,
                    "filed": filing.filed,
                },
            )
            report.unresolved.append(f"{name} -> {supplier_ticker}")
        else:
            relate_db.query(
                conn,
                f"""
                MERGE (v:{SUPPLIER} {{name: $name}})
                WITH v
                MATCH (cust:{STOCK} {{ticker: $customer}})
                MERGE (v)-[r:SUPPLIES_TO]->(cust)
                SET r.discovered = true,
                    r.source = coalesce(r.source, $form),
                    r.accession = $accession,
                    r.filed = $filed
                """,
                {
                    "name": name,
                    "customer": report.ticker,
                    "form": filing.form,
                    "accession": filing.accession,
                    "filed": filing.filed,
                },
            )

    for name in report.subsidiaries:
        relate_db.query(
            conn,
            f"""
            MERGE (sub:{SUBSIDIARY} {{name: $name}})
            WITH sub
            MATCH (s:{STOCK} {{ticker: $ticker}})
            MERGE (s)-[r:OWNS]->(sub)
            SET r.source = coalesce(r.source, $form),
                r.accession = $accession
            """,
            {"name": name, "ticker": report.ticker, "form": filing.form,
             "accession": filing.accession},
        )

    report.wrote = True
    report.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return report


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+", help="tickers to fetch, e.g. AAOI")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--dry-run", action="store_true", help="fetch but do not write")
    parser.add_argument("--no-model", action="store_true",
                        help="deterministic extraction only (no LLM fallback)")
    args = parser.parse_args(argv)

    if not config.SEC_CONTACT_EMAIL:
        print("SEC_CONTACT_EMAIL is not set; EDGAR requires a contact address.")
        return 2

    conn = relate_db.connect()
    relate_db.ensure_schema(conn)
    reports: list[FastReport] = []
    try:
        for ticker in args.tickers:
            report = ingest(
                conn,
                ticker,
                write=not args.dry_run,
                allow_model=not args.no_model,
            )
            reports.append(report)
            if args.json:
                continue
            print(
                f"\n=== {report.ticker} ({report.company_name or '?'}) "
                f"{report.form or ''} {report.filed or ''}"
            )
            print(
                f"    download {report.document_bytes/1024:.0f} KB"
                f"{' (truncated)' if report.document_truncated else ''}"
                f" | text {report.text_chars:,} chars"
                f" | section {report.section_chars:,} chars"
                f" | {report.elapsed_ms} ms"
                f" | {report.tokens} tokens"
            )
            print(f"    suppliers ({len(report.suppliers)}): {report.suppliers}")
            if report.evidence:
                print(f"    evidence: {report.evidence}")
            if report.cached:
                print("    (de-tagged text from cache - no download this run)")
            if report.unresolved:
                print(f"    resolved to tickers: {report.unresolved}")
            if report.subsidiaries:
                print(f"    subsidiaries ({len(report.subsidiaries)}): "
                      f"{report.subsidiaries[:8]}")
            for note in report.notes:
                print(f"    note: {note}")
            if report.wrote:
                print("    written to graph")
    finally:
        conn.close()

    if args.json:
        print(json.dumps([report.to_dict() for report in reports], indent=2))
    # Reading the filing IS the success condition: "this company's 10-K names no
    # suppliers" is a real answer, not a failure.
    return 0 if all(report.form for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
