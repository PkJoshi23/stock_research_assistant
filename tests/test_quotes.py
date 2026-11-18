"""One quote path.

Before this module existed, the live answer tried ``fast_info`` then ``.info``
while ingestion tried the opposite order - so the same ticker could produce two
different numbers depending on which code path asked. These tests pin the single
documented order and the "return None, never raise" contract.
"""

from __future__ import annotations

import types

import pytest

from src import config, quotes


class _Endpoint:
    """Minimal yfinance stand-in with independently failing endpoints."""

    def __init__(self, *, fast=None, info=None, fast_raises=False, info_raises=False):
        self._fast = fast
        self._info = info
        self._fast_raises = fast_raises
        self._info_raises = info_raises

    @property
    def fast_info(self):
        if self._fast_raises:
            raise RuntimeError("fast_info unavailable")
        return self._fast or {}

    @property
    def info(self):
        if self._info_raises:
            raise RuntimeError("info unavailable")
        return self._info or {}


@pytest.fixture(autouse=True)
def _no_cache_side_effect(monkeypatch):
    monkeypatch.setattr(config, "configure_yfinance_cache", lambda: None)


def _install(monkeypatch, endpoint) -> None:
    fake = types.SimpleNamespace(Ticker=lambda ticker: endpoint)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake)


def test_fast_info_is_preferred(monkeypatch):
    _install(
        monkeypatch,
        _Endpoint(fast={"lastPrice": 10.0, "currency": "USD"}, info={"currentPrice": 99.0}),
    )
    quote = quotes.fetch("NVDA")
    assert quote is not None
    assert quote.price == 10.0
    assert quote.source == "fast_info"


def test_info_is_the_documented_fallback(monkeypatch):
    _install(monkeypatch, _Endpoint(fast_raises=True, info={"currentPrice": 42.5, "currency": "USD"}))
    quote = quotes.fetch("NVDA")
    assert quote.price == 42.5
    assert quote.source == "info"


def test_regular_market_price_is_used_when_current_price_is_absent(monkeypatch):
    _install(monkeypatch, _Endpoint(fast_raises=True, info={"regularMarketPrice": 7.25}))
    assert quotes.fetch("NVDA").price == 7.25


def test_no_price_anywhere_returns_none(monkeypatch):
    _install(monkeypatch, _Endpoint(fast_raises=True, info={}))
    assert quotes.fetch("NVDA") is None


def test_both_endpoints_failing_returns_none_instead_of_raising(monkeypatch):
    _install(monkeypatch, _Endpoint(fast_raises=True, info_raises=True))
    assert quotes.fetch("NVDA") is None


def test_quote_records_its_provenance(monkeypatch):
    _install(monkeypatch, _Endpoint(fast={"lastPrice": 1.5, "currency": "EUR"}))
    quote = quotes.fetch("SAP")
    assert quote.ticker == "SAP"
    assert quote.currency == "EUR"
    assert quote.at  # ISO timestamp
    assert "T" in quote.at
