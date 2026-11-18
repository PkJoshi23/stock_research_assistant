"""One way to fetch a current quote.

There were two implementations of "get the price" with **opposite fallback
order**: the live-answer path tried ``fast_info`` then ``.info``, while stock
ingestion tried ``.info`` then ``fast_info``. Both are plausible, which is the
problem - the same ticker could produce two different numbers depending on which
code path asked, and nothing in the outputs said which path had won.

This module makes the order explicit and single: ``fast_info`` (cheap, one
request) first, ``.info`` (heavier, richer) as the fallback. It also owns the
``configure_yfinance_cache()`` call that three entry points used to remember
individually.

Scope note: this returns a *current* quote. Nothing here is persisted, which is
what keeps the graph free of price history.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from src import config


@dataclass(frozen=True)
class Quote:
    """A single point-in-time quote. Deliberately not a history."""

    ticker: str
    price: float
    currency: str
    source: str  # "fast_info" | "info" - which endpoint answered
    at: str  # ISO timestamp of the fetch


def fetch(ticker: str) -> Quote | None:
    """Return the current quote for ``ticker``, or ``None`` if unavailable.

    Returns ``None`` rather than raising: callers render a "couldn't retrieve"
    message, and an exception here would take down an otherwise good answer.

    yfinance is imported inside the function on purpose - it is a heavy import
    and only this path needs it.
    """
    # Must run before the first yf.Ticker(): yfinance keeps SQLite caches in a
    # user-level directory that is not writable here and fails with the opaque
    # "OperationalError: unable to open database file".
    config.configure_yfinance_cache()

    import yfinance as yf

    stock = yf.Ticker(ticker)
    price: float | None = None
    currency = ""
    source = ""

    try:
        fast = stock.fast_info
        price = fast["lastPrice"]
        currency = fast["currency"] or ""
        source = "fast_info"
    except Exception:  # noqa: BLE001 - fall through to the heavier endpoint
        price = None

    if price is None:
        try:
            info = stock.info or {}
        except Exception:  # noqa: BLE001 - network failure on both endpoints
            return None
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        currency = info.get("currency") or ""
        source = "info"

    if price is None:
        return None

    return Quote(
        ticker=ticker,
        price=float(price),
        currency=str(currency or ""),
        source=source,
        at=dt.datetime.now().isoformat(timespec="seconds"),
    )
