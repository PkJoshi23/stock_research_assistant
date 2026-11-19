"""Deterministic coverage gate: is the question about a company we track?

The pipeline is happy to spend three model calls answering "who supplies to
AAOI?" and then say "no data". This module decides, with no model and no
network, whether the question names a company at all - and if it names only
companies the graph does not contain, the agent can answer immediately:

    I don't track AAOI (Applied Optoelectronics, Inc.). My graph covers 18
    companies: ...

Two signals, both local:

* **tracked** - ``entities.resolve(..., policy="scan")`` already knows every
  name, ticker and alias in the graph;
* **untracked** - SEC's cached registrant map (10,400 tickers), used only to
  recognise a *ticker-shaped* token: upper-case always, lower-case only when it
  is at least four characters (so "aaoi" is found but "on"/"so"/"ai" are not).

The gate never *blocks* a question that mentions a tracked company, and never
blocks a thematic question that names no company at all. False positives are
tolerated in one direction only: the cost of refusing is a helpful message,
while the cost of proceeding is a wasted pipeline run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src import entities, ticker_directory

#: Words that are also tickers but are far more often ordinary English. Without
#: this, "what does the market say about all of this?" would ask about ALL.
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "who", "whom",
    "which", "when", "where", "why", "how", "are", "was", "were", "been", "being",
    "have", "has", "had", "does", "did", "can", "could", "should", "would",
    "will", "may", "might", "must", "not", "its", "their", "there", "these",
    "those", "they", "them", "then", "than", "also", "about", "into", "over",
    "under", "after", "before", "between", "during", "through", "against",
    "because", "while", "some", "any", "all", "most", "more", "less", "much",
    "many", "such", "only", "own", "same", "other", "both", "each", "few",
    "very", "just", "now", "today", "latest", "current", "recent",
    "stock", "stocks", "price", "prices", "trading", "quote", "supplier",
    "suppliers", "supply", "supplies", "competitor", "competitors", "sector",
    "executive", "executives", "board", "subsidiary", "subsidiaries", "graph",
    "data", "company", "companies", "business", "market", "cap", "share",
    "shares", "earnings", "report", "filing", "sec", "chip", "chips", "ai",
    "open",
    "grab", "key", "play", "real", "live", "work", "best", "fast", "next",
    "move", "cost", "cash", "fund", "bank", "care", "plan", "home", "safe",
    "pass", "ever", "core", "pure", "true", "shop", "star", "wave", "snow",
    "zoom", "unit", "team", "peak", "well", "gold", "iron", "fire", "free",
    "hope", "land", "life", "line", "mark", "mind", "north", "note", "page",
    "park", "path", "pick", "pine", "pipe", "plus", "post", "prime", "rain",
    "rare", "rich", "rise", "road", "rock", "root", "rose", "rule", "rush",
    "sage", "sale", "salt", "save", "seat", "seed", "sell", "ship", "side",
    "sign", "silk", "site", "size", "skin", "slow", "soft", "soil", "sole",
    "song", "soul", "sound", "south", "spin", "spot", "stay", "stem", "step",
    "stone", "stop", "store", "storm", "story", "sun", "sure", "tail", "take",
    "tale", "talk", "tall", "task", "tell", "tend", "term", "test", "text",
    "thin", "tide", "tier", "tile", "time", "tiny", "tire", "toll", "tone",
    "tool", "tour", "town", "tree", "trend", "trip", "tune", "turn", "twin",
    "type", "upon", "used", "user", "vast", "view", "vote", "wage", "wait",
    "wake", "walk", "wall", "want", "warm", "wash", "weak", "wear", "week",
    "west", "wide", "wife", "wild", "wind", "wine", "wing", "wire", "wise",
    "wish", "wood", "word", "yard", "year", "your", "zero",
}

#: A ticker-shaped word: 1-5 characters, optional dot/dash class suffix.
_TOKEN = re.compile(r"\b([A-Za-z][A-Za-z0-9]{0,5})\b")

_PRICE_HINTS = re.compile(
    r"\b(?:price|trading|trade|quote|worth|cost|valuation)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class Coverage:
    """What the question is about, and whether the graph can answer it."""

    tracked: list[str] = field(default_factory=list)
    untracked: list[tuple[str, str]] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        """Only untracked companies named, and nothing tracked to fall back on."""
        return bool(self.untracked) and not self.tracked

    def tickers(self) -> list[str]:
        return [*self.tracked, *(ticker for ticker, _ in self.untracked)]


def looks_like_price_question(question: str) -> bool:
    return bool(_PRICE_HINTS.search(question))


def _ticker_tokens(question: str) -> list[str]:
    """Ticker-shaped tokens, using case and the registrant map as the filter."""
    try:
        known = {str(entry.get("ticker", "")).upper() for entry in ticker_directory.load_directory()}
    except Exception:  # noqa: BLE001 - no registry available: gate is optional
        return []

    found: list[str] = []
    for match in _TOKEN.finditer(question):
        token = match.group(1)
        upper = token.upper()
        if upper not in known:
            continue
        if token.lower() in _STOPWORDS:
            continue
        if token.isupper():
            found.append(upper)
        elif len(token) >= 4:
            # Lower-case only for longer tickers: "aaoi" yes, "on"/"so"/"ai" no.
            found.append(upper)
    return list(dict.fromkeys(found))


def assess(question: str, companies: list[dict]) -> Coverage:
    """Classify the companies a question names as tracked or untracked."""
    tracked = [match.ticker for match in entities.resolve(question, companies, policy="scan")]
    tracked = list(dict.fromkeys(tracked))

    names = {company["ticker"].upper(): company.get("name") or "" for company in companies}
    untracked: list[tuple[str, str]] = []
    for ticker in _ticker_tokens(question):
        if ticker in names:
            continue  # resolved as tracked above (or a name we hold)
        untracked.append((ticker, _registered_name(ticker)))

    # A tracked mention anywhere means the graph can still answer.
    return Coverage(tracked=tracked, untracked=[] if tracked else untracked)


def _registered_name(ticker: str) -> str:
    try:
        for entry in ticker_directory.load_directory():
            if str(entry.get("ticker", "")).upper() == ticker:
                return str(entry.get("title") or "")
    except Exception:  # noqa: BLE001
        pass
    return ""


def refusal_text(coverage: Coverage, tracked_tickers: list[str]) -> str:
    """The message shown when the question is about an untracked company."""
    described = ", ".join(
        f"**{ticker}**" + (f" ({name.title()})" if name else "")
        for ticker, name in coverage.untracked[:3]
    )
    listed = ", ".join(sorted(tracked_tickers)[:14])
    return (
        f"I don't track {described}. "
        f"My graph covers {len(tracked_tickers)} companies: {listed}.\n\n"
        "I can fetch its latest SEC filing and add it in about 6 seconds - "
        "its named suppliers included."
    )
