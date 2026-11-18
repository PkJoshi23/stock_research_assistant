"""Shared entity resolution: tracked-stock lookup and mention scanning.

Two ingestion paths need the same answer to "which tracked company is this?":

  * the SEC loader resolves a supplier named in a 10-K ("TSMC") onto the
    existing ``Stock:TSM`` node instead of inventing a parallel entity that the
    agent would never join to;
  * the news loader scans an article's title + summary for *every* watchlist
    company it names, so one story about an Nvidia/TSMC supply deal links to
    both stocks instead of only the one that was searched for.

Both live here so they cannot drift apart. When the two paths each carried
their own matcher, the same company ended up with two identities and queries
silently returned half an answer.

Two match policies, on purpose, because the inputs differ
---------------------------------------------------------
* :func:`resolve_tracked_stock` - conservative whole-name matching for
  model-extracted names. A 10-K line about "cloud infrastructure providers"
  must not resolve to a random stock.
* :func:`find_mentioned_tickers` - scanning free text. Recall matters there
  (a headline writes "TSMC", not the legal name), but a substring hit like
  "AMD" inside another word must never become an edge.

Edge policy for the news graph lives with its caller
(``src/ingestion/ingest_connected_news.py``). This module only reads the graph;
it never writes to it.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from src import relate_db
from src.schema import STOCK

#: Shortest acronym-style surface form that is safe to match.
#:
#: One-character tickers exist (Ford is ``F``) but a bare ``F`` cannot be told
#: apart from a stray capital in prose, so single characters are not matched as
#: tickers. Ford is still found through its name/alias ("Ford"), which is how
#: news copy writes it anyway.
MIN_ACRONYM_LENGTH = 2

#: Shortest word-style surface form that is safe to match. Two-letter words are
#: mostly noise in a news context; every alias in the curated watchlist is
#: three characters or longer, so this only rejects malformed data.
MIN_WORD_LENGTH = 3


# -----------------------------------------------------------------------------
# Graph reads
# -----------------------------------------------------------------------------


def load_companies(target) -> list[dict]:
    """Return every tracked company as ``{ticker, name, aliases}``.

    ``target`` is either a raw connection (ingestion) or a ``GraphSession``
    (the agent). Accepting both is the point: the agent used to be unable to
    reuse this loader, so it grew its own matching instead.

    Sorted by ticker rather than in Cypher: the scan order decides the order of
    ``MENTIONED_IN`` edges written by the news loader, and a deterministic order
    is what makes re-running ingestion reproducible.
    """
    rows = relate_db.query(
        target,
        f"MATCH (s:{STOCK}) RETURN s.ticker AS ticker, s.name AS name, s.aliases AS aliases",
    )
    companies: list[dict] = []
    for row in rows:
        ticker = (row["ticker"] or "").strip()
        if not ticker:
            # A Stock without a ticker cannot be matched or linked; skip it
            # rather than letting an empty string match every article.
            continue
        aliases = row["aliases"] or []
        if not isinstance(aliases, list):
            aliases = [aliases]
        companies.append(
            {
                "ticker": ticker,
                "name": row["name"] or "",
                "aliases": [str(alias) for alias in aliases if alias],
            }
        )
    companies.sort(key=lambda company: company["ticker"])
    return companies


def surface_forms(company: dict) -> list[str]:
    """Every string a company answers to: ticker, legal name and aliases.

    De-duplicated on the exact spelling only. Casing is *not* folded here on
    purpose: :func:`_mention_pattern` derives the match policy from the shape of
    the form, so ``META`` (an acronym, matched case-sensitively) and ``Meta``
    (a name, matched case-insensitively) are two different patterns. Collapsing
    them case-insensitively would silently lose the spelling that can match
    "Meta" in running text.
    """
    raw = [
        company.get("ticker") or "",
        company.get("name") or "",
        *(company.get("aliases") or []),
    ]
    forms: list[str] = []
    seen: set[str] = set()
    for value in raw:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        forms.append(text)
    return forms


# -----------------------------------------------------------------------------
# Matching
# -----------------------------------------------------------------------------
#
# Two policies exist because the inputs differ, and that difference is real:
#
#   * ``strict`` - a name extracted from a filing ("TSMC"). Whole-string
#     comparison only; "cloud infrastructure providers" must not resolve to a
#     random stock.
#   * ``scan``   - free text (a headline, a question). Boundary-aware matching,
#     ranked by position.
#
# They share one interface so the graph is read once and the rules live in one
# place. Callers pick a policy; they never pick an implementation.


@dataclass(frozen=True)
class Match:
    """One company found in a piece of text."""

    ticker: str
    form: str  # the surface form that matched ("TSMC", "Nvidia", "NVDA")
    position: int  # first character offset, or 0 for whole-string matches
    policy: str  # "strict" | "scan"


def _loose(text: str) -> str:
    """Normalise for whole-name comparison: lowercase, punctuation to spaces.

    Punctuation has to go: a filing writes "Micron Technology, Inc" while the
    graph stores "Micron Technology, Inc." - with the trailing period intact the
    boundary-aware containment never matches, and the crawler keeps a duplicate
    vendor node next to the real stock.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def _strict(text: str, companies: list[dict], exclude: set[str]) -> list[Match]:
    """Whole-name containment for model-extracted names."""
    needle = _loose(text)
    if len(needle) < 3:
        return []

    matches: list[Match] = []
    padded_needle = f" {needle} "
    for company in companies:
        ticker = company["ticker"]
        if ticker.upper() in exclude:
            continue
        for candidate in surface_forms(company):
            value = _loose(candidate)
            if not value:
                continue
            padded_value = f" {value} "
            if padded_value == padded_needle:
                return [Match(ticker, candidate, 0, "strict")]
            shorter, longer = (
                (padded_value, padded_needle)
                if len(value) <= len(needle)
                else (padded_needle, padded_value)
            )
            # Token boundaries, not substrings: the 4-character registrant
            # "TRON" must not match the supplier "Wistron".
            if len(shorter.strip()) >= 4 and shorter in longer:
                matches.append(Match(ticker, candidate, 0, "strict"))
                break
    return matches[:1]


def _scan(text: str, companies: list[dict], exclude: set[str]) -> list[Match]:
    """Boundary-aware scan of free text, ordered by first match position."""
    if not text:
        return []

    first: dict[str, Match] = {}
    for company in companies:
        ticker = company["ticker"]
        if not ticker or ticker.upper() in exclude:
            continue
        for form in surface_forms(company):
            pattern = _mention_pattern(form)
            if pattern is None:
                continue
            match = pattern.search(text)
            if match is None:
                continue
            existing = first.get(ticker)
            if existing is None or match.start() < existing.position:
                first[ticker] = Match(ticker, form, match.start(), "scan")
    return sorted(first.values(), key=lambda item: (item.position, item.ticker))


def resolve(
    text: str | None,
    companies: list[dict],
    *,
    policy: str = "scan",
    exclude: tuple[str, ...] | list[str] = (),
) -> list[Match]:
    """Find the tracked companies named in ``text``.

    A single interface over both policies. It never reads the graph - callers
    pass ``companies`` in from :func:`load_companies`, so one read serves however
    many texts they need to check (ingestion checks hundreds of headlines per
    run; the agent checks one question).
    """
    if not text:
        return []
    excluded = {ticker.upper() for ticker in exclude}
    if policy == "scan":
        return _scan(text, companies, excluded)
    if policy == "strict":
        return _strict(text, companies, excluded)
    raise ValueError(f"unknown policy {policy!r}; expected 'scan' or 'strict'")


def best_match(
    text: str,
    companies: list[dict],
    *,
    hint: tuple[str, ...] | list[str] = (),
) -> str | None:
    """Pick the single company a question is about.

    Order: a model-supplied hint that is actually tracked, then a scan of the
    text. Hints that are not tracked are still honoured (the caller may want to
    say "I don't track that"), matching the behaviour the agent had before.

    Returns ``None`` rather than guessing: a wrong ticker is worse than none.
    """
    known = {company["ticker"].upper() for company in companies}
    for candidate in hint or ():
        ticker = str(candidate).strip().upper()
        if ticker and ticker in known:
            return ticker
    if hint:
        return str(hint[0]).strip().upper() or None

    # A bare ticker is matched exactly, not by scanning. One-character tickers
    # (Ford is "F") are excluded from scanning because a lone "F" matches far too
    # much prose - but "F" typed alone is unambiguous.
    bare = text.strip().rstrip("?.!").strip().upper()
    for company in companies:
        if company["ticker"].upper() == bare:
            return company["ticker"]

    matches = _scan(text, companies, set())
    return matches[0].ticker if matches else None


def resolve_tracked_stock(target, name: str) -> str | None:
    """Map a filing's company name onto a tracked ticker, if one matches.

    Conservative by policy: exact ticker, exact alias, or a containment where the
    shorter side is at least four characters.
    """
    matches = resolve(name, load_companies(target), policy="strict")
    return matches[0].ticker if matches else None


# -----------------------------------------------------------------------------
# Scanning free text (news)
# -----------------------------------------------------------------------------


@functools.lru_cache(maxsize=2048)
def _mention_pattern(form: str) -> re.Pattern[str] | None:
    """Compile the word-boundary pattern for one surface form, or ``None``.

    ``(?<!\\w)`` / ``(?!\\w)`` rather than ``\\b``: a legal name can end in a
    full stop ("Advanced Micro Devices, Inc."), and ``\\b`` after that dot does
    not match when a space follows, so the name would never be found.

    Case sensitivity is chosen by shape:

    * acronyms (all upper case - tickers, ``TSMC``, ``AWS``) are matched
      case-sensitively. Prose writes them in capitals, while a lower-case run
      like ``meta`` or ``intel`` is ordinary English far more often than it is
      a ticker;
    * everything else is matched case-insensitively so "Nvidia", "NVIDIA" and
      "nvidia" all resolve.

    The trade-off is documented rather than hidden: a lower-case English word
    that equals a company name ("meta-analysis") will link that company. In
    market copy that is rare, and the alternative - demanding an exact
    capitalisation for every name - loses real mentions in summaries that are
    written in lower case.
    """
    text = form.strip()
    if not text:
        return None
    acronym = text.isupper()
    minimum = MIN_ACRONYM_LENGTH if acronym else MIN_WORD_LENGTH
    if len(text) < minimum:
        return None
    flags = 0 if acronym else re.IGNORECASE
    return re.compile(rf"(?<!\w){re.escape(text)}(?!\w)", flags)


def find_mentioned_tickers(
    text: str | None,
    companies: list[dict],
    *,
    exclude: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """Return every tracked company named in ``text``, in a stable order.

    A thin projection of :func:`resolve` for callers that only need tickers.
    ``exclude`` holds tickers whose link is written elsewhere with a stronger
    provenance (the news loader excludes the ticker it searched for, so
    ``via='search'`` is not demoted to ``via='mention'``).
    """
    return [
        match.ticker
        for match in resolve(text, companies, policy="scan", exclude=exclude)
    ]
