"""Name -> ticker resolution for discovered companies.

The authority is SEC's registrant map; these tests use a fixture shaped like it,
including the duplicate-ticker problem (TSM/TSMWF, ORCL/ORCL-PD) that forces a
ranking rule.
"""

from __future__ import annotations

import json

import pytest

from src import config, ticker_directory

ENTRIES = [
    {"ticker": "NVDA", "title": "NVIDIA CORP"},
    {"ticker": "TSM", "title": "TAIWAN SEMICONDUCTOR MANUFACTURING CO LTD"},
    {"ticker": "TSMWF", "title": "TAIWAN SEMICONDUCTOR MANUFACTURING CO LTD"},
    {"ticker": "ORCL", "title": "ORACLE CORP"},
    {"ticker": "ORCL-PD", "title": "ORACLE CORP"},
    {"ticker": "MU", "title": "MICRON TECHNOLOGY INC"},
    {"ticker": "VRT", "title": "VERTIV HOLDINGS CO"},
    {"ticker": "MRVL", "title": "Marvell Technology, Inc."},
    # A short registrant whose name is a substring of a real supplier name -
    # this resolved "Wistron Corporation" to TRON before boundaries were added.
    {"ticker": "TRON", "title": "TRON"},
]


@pytest.fixture
def directory() -> ticker_directory.TickerDirectory:
    return ticker_directory.TickerDirectory(ENTRIES)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Micron Technology, Inc.", "MU"),
        ("NVIDIA CORP", "NVDA"),
        ("Marvell Technology, Inc.", "MRVL"),
        ("Vertiv", "VRT"),  # containment against "vertiv holdings"
        ("Oracle", "ORCL"),
        # Duplicate tickers: the plain listing must win over the OTC/class variant.
        ("Taiwan Semiconductor Manufacturing Company Limited", "TSM"),
    ],
)
def test_lookup_resolves_realistic_names(directory, name, expected):
    assert directory.lookup(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "cloud infrastructure providers",
        "various suppliers",
        "AI",
        "",
    ],
)
def test_lookup_refuses_to_guess(directory, name):
    assert directory.lookup(name) is None


def test_legal_suffixes_are_ignored(directory):
    assert directory.lookup("Micron Technology Incorporated") == "MU"


def test_containment_respects_token_boundaries(directory):
    """A short registrant name must not match inside a longer word.

    Real false positive: "Wistron Corporation" resolved to the registrant "TRON"
    because "tron" is a substring of "wistron".
    """
    assert directory.lookup("Wistron Corporation") is None
    # ... while a genuine token match still works.
    assert directory.lookup("Vertiv Holdings") == "VRT"


def test_duplicate_ranking_prefers_plain_tickers(directory):
    assert directory.lookup("Oracle Corporation") == "ORCL"
    assert directory.lookup("Taiwan Semiconductor") == "TSM"


def test_directory_is_built_from_the_registrant_map(directory):
    assert directory.lookup("Nvidia") == "NVDA"


# -- fetching / caching ------------------------------------------------------


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture
def cached_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SEC_CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "SEC_CONTACT_EMAIL", "test@example.com")
    return tmp_path


def test_directory_is_cached_after_first_fetch(cached_dir, monkeypatch):
    calls: list[str] = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return _Response(
            {
                "0": {
                    "cik_str": 1045810,
                    "ticker": "NVDA",
                    "title": "NVIDIA CORP",
                }
            }
        )

    monkeypatch.setattr("requests.get", fake_get)

    first = ticker_directory.load_directory()
    second = ticker_directory.load_directory()

    assert first == second == [
        {"ticker": "NVDA", "title": "NVIDIA CORP", "cik": "0001045810"}
    ]
    assert len(calls) == 1  # second call read the cache
    assert (cached_dir / "company_tickers.json").exists()


def test_stale_cache_without_cik_is_refetched(cached_dir, monkeypatch):
    """The fast SEC path needs CIK; an older cache must not silently lack it."""
    path = cached_dir / "company_tickers.json"
    path.write_text(json.dumps([{"ticker": "NVDA", "title": "NVIDIA CORP"}]), encoding="utf-8")

    monkeypatch.setattr(
        "requests.get",
        lambda *a, **k: _Response(
            {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"}}
        ),
    )

    assert ticker_directory.load_directory()[0]["cik"] == "0001045810"


def test_missing_contact_email_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SEC_CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "SEC_CONTACT_EMAIL", "")

    with pytest.raises(RuntimeError, match="SEC_CONTACT_EMAIL"):
        ticker_directory.load_directory()


def test_corrupt_cache_is_refetched(cached_dir, monkeypatch):
    path = cached_dir / "company_tickers.json"
    path.write_text("{not json", encoding="utf-8")

    monkeypatch.setattr(
        "requests.get",
        lambda *a, **k: _Response(
            {"0": {"cik_str": 723125, "ticker": "MU", "title": "MICRON TECHNOLOGY INC"}}
        ),
    )

    assert ticker_directory.load_directory() == [
        {"ticker": "MU", "title": "MICRON TECHNOLOGY INC", "cik": "0000723125"}
    ]
