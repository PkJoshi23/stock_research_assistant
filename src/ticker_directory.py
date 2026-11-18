"""Name -> ticker resolution for *discovered* companies.

``src/entities.py`` answers "is this one of the companies I already track?". The
crawler needs the other direction: a 10-K says "Micron Technology, Inc." and the
crawler has to decide whether that is a listed company it can keep expanding.

Source: SEC's own registrant map (``company_tickers.json``, ~10,400 companies,
no key required, cached after the first fetch). It is the right authority here
because the names being resolved come *from* SEC filings - the spelling in the
map is the spelling in the filing.

Duplicate tickers are common in that file (``TSM`` and ``TSMWF`` are the same
company; ``ORCL`` and ``ORCL-PD`` likewise). ``_rank`` picks the primary listing:
plain tickers first, then the shortest, then alphabetical.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src import config

#: Legal-form words dropped before comparing names.
_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd",
    "limited", "llc", "plc", "nv", "sa", "ag", "se", "lp", "holdings",
    "group", "the",
}

_NON_WORD = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")

#: Prefer a plain ticker: no dash, no dot, no trailing class/F suffix.
_PLAIN_TICKER = re.compile(r"^[A-Z]{1,5}$")


def _normalise(name: str) -> str:
    """Lower-case, strip punctuation and trailing legal forms."""
    text = _NON_WORD.sub(" ", name.lower())
    words = [word for word in _WS.sub(" ", text).split() if word]
    while words and words[-1] in _LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words)


def _rank(entries: list[dict]) -> list[dict]:
    """Primary listings first: plain ticker, then shorter, then alphabetical."""
    return sorted(
        entries,
        key=lambda entry: (
            0 if _PLAIN_TICKER.match(entry["ticker"]) else 1,
            len(entry["ticker"]),
            entry["ticker"],
        ),
    )


def cache_path() -> Path:
    return config.SEC_CACHE_DIR / "company_tickers.json"


def load_directory(*, refresh: bool = False) -> list[dict]:
    """Return ``[{ticker, title}]``, fetching SEC's map once and caching it."""
    path = cache_path()
    if path.exists() and not refresh:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass  # fall through to a refetch

    if not config.SEC_CONTACT_EMAIL:
        raise RuntimeError(
            "SEC_CONTACT_EMAIL is not set; EDGAR requires a contact address. "
            "Set it in .env (see .env.example)."
        )

    import requests

    response = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={
            "User-Agent": f"{config.SEC_COMPANY_NAME} {config.SEC_CONTACT_EMAIL}",
            "Accept-Encoding": "gzip, deflate",
        },
        timeout=config.HTTP_TIMEOUT_S,
    )
    response.raise_for_status()

    payload = response.json()
    entries = [
        {"ticker": str(item["ticker"]).upper(), "title": str(item["title"])}
        for item in payload.values()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")
    return entries


class TickerDirectory:
    """Look up a company name and get the most likely listed ticker."""

    def __init__(self, entries: list[dict]) -> None:
        self._by_name: dict[str, list[dict]] = {}
        for entry in entries:
            key = _normalise(entry["title"])
            if key:
                self._by_name.setdefault(key, []).append(entry)

    @classmethod
    def load(cls, *, refresh: bool = False) -> "TickerDirectory":
        return cls(load_directory(refresh=refresh))

    def lookup(self, name: str) -> str | None:
        """Resolve a filing's company name to a ticker, or ``None``.

        Two strategies, both conservative:

        1. exact match after normalisation ("Micron Technology, Inc." ->
           "micron technology" == "MICRON TECHNOLOGY INC");
        2. containment either way, minimum four characters ("Vertiv" inside
           "vertiv holdings").

        Anything less certain returns ``None`` so the caller can keep the name as
        an unlisted supplier rather than inventing a wrong ticker.
        """
        needle = _normalise(name)
        if len(needle) < 3:
            return None

        exact = self._by_name.get(needle)
        if exact:
            return _rank(exact)[0]["ticker"]

        candidates: list[dict] = []
        padded_needle = f" {needle} "
        for key, entries in self._by_name.items():
            padded_key = f" {key} "
            shorter, longer = (
                (padded_needle, padded_key)
                if len(needle) <= len(key)
                else (padded_key, padded_needle)
            )
            # Token boundaries, not substrings: "tron" must not match inside
            # "wistron" (a real false positive that resolved Wistron -> TRON).
            if len(shorter.strip()) >= 4 and shorter in longer:
                candidates.extend(entries)
        if not candidates:
            return None
        return _rank(candidates)[0]["ticker"]
