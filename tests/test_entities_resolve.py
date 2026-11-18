"""One interface for "which tracked company is this text about?".

Two policies, deliberately: ``strict`` for names extracted from filings,
``scan`` for free text. What these tests pin is that both live behind one
function, read the graph once per caller, and stay deterministic.
"""

from __future__ import annotations

import pytest

from src import entities, relate_db


@pytest.fixture
def companies(seeded):
    return entities.load_companies(seeded)


@pytest.fixture
def companies_with_ford(seeded):
    """Ford has a one-character ticker - the case that exposed the old bug."""
    seeded.run(
        "MERGE (s:Stock {ticker: 'F'}) "
        "SET s.name = 'Ford Motor Company', s.aliases = ['Ford']"
    )
    return entities.load_companies(seeded)


def test_load_companies_accepts_a_session(seeded):
    """The reason the agent could not reuse this loader before: it demanded a
    raw connection, so the agent grew its own matcher."""
    assert len(entities.load_companies(seeded)) == 4


def test_load_companies_accepts_a_raw_connection(conn):
    conn.execute("CREATE (:Stock {ticker: 'ZZZ', name: 'Zeta Corp'})")
    assert [c["ticker"] for c in entities.load_companies(conn)] == ["ZZZ"]


def test_query_adapter_dispatches_for_a_session(seeded):
    assert relate_db.query(seeded, "RETURN 1 AS one")[0]["one"] == 1


def test_query_adapter_dispatches_for_a_connection(conn):
    # Separate test on purpose: a session and a raw connection cannot share a
    # store (Relate grants one writable lease), so the adapter's two branches
    # are checked against two stores.
    assert relate_db.query(conn, "RETURN 2 AS two")[0]["two"] == 2


def test_scan_finds_a_ticker_acronym_case_sensitively(companies):
    matches = entities.resolve("NVDA is up today", companies)
    assert [m.ticker for m in matches] == ["NVDA"]
    assert matches[0].policy == "scan"


def test_scan_does_not_match_a_lowercase_acronym(companies):
    assert entities.resolve("nvda is up today", companies) == []


def test_scan_matches_names_case_insensitively(companies):
    assert [m.ticker for m in entities.resolve("nvidia beats estimates", companies)] == ["NVDA"]


def test_scan_uses_boundaries(companies):
    # "AMD" inside another word must not become an edge; "TSMC" inside a sentence must.
    assert entities.resolve("the AMDRAM standard", companies) == []
    assert [m.ticker for m in entities.resolve("TSMC raises capex", companies)] == ["TSM"]


def test_scan_orders_by_position(companies):
    matches = entities.resolve("Nvidia and AMD both rose", companies)
    assert [m.ticker for m in matches] == ["NVDA", "AMD"]


def test_scan_honours_exclude(companies):
    matches = entities.resolve("Nvidia and AMD", companies, exclude=["NVDA"])
    assert [m.ticker for m in matches] == ["AMD"]


def test_strict_matches_a_whole_legal_name(companies):
    matches = entities.resolve(
        "Taiwan Semiconductor Manufacturing Company Limited", companies, policy="strict"
    )
    assert [m.ticker for m in matches] == ["TSM"]


def test_strict_rejects_generic_phrases(companies):
    assert entities.resolve("cloud infrastructure providers", companies, policy="strict") == []


def test_strict_respects_token_boundaries(seeded):
    """A short registrant must not match inside a longer supplier name.

    Real false positive: the crawler resolved "Wistron Corporation" to the
    four-letter registrant TRON, because "tron" is a substring of "wistron".
    """
    seeded.run("MERGE (s:Stock {ticker: 'TRON'}) SET s.name = 'TRON'")
    companies = entities.load_companies(seeded)
    assert entities.resolve("Wistron Corporation", companies, policy="strict") == []


def test_strict_requires_three_characters(companies):
    assert entities.resolve("AI", companies, policy="strict") == []


def test_unknown_policy_is_a_loud_error(companies):
    with pytest.raises(ValueError):
        entities.resolve("Nvidia", companies, policy="fuzzy")


def test_best_match_prefers_a_tracked_hint(companies):
    assert entities.best_match("anything at all", companies, hint=["AMD"]) == "AMD"


def test_best_match_honours_an_untracked_hint(companies):
    """Matching the old behaviour: a valid-looking but un-ingested ticker is
    still reported, so the caller can say "I don't track that"."""
    assert entities.best_match("anything", companies, hint=["ZZZZ"]) == "ZZZZ"


def test_best_match_falls_back_to_scanning(companies):
    assert entities.best_match("who supplies Nvidia?", companies) == "NVDA"


def test_best_match_returns_none_rather_than_guessing(companies):
    assert entities.best_match("what is the weather in Tokyo", companies) is None


def test_best_match_accepts_a_bare_ticker(companies_with_ford):
    """Ford's ticker is one character, so scanning excludes it as noise - but a
    question that is *just* "F" is unambiguous."""
    assert entities.best_match("F", companies_with_ford) == "F"
    assert entities.best_match("nvda?", companies_with_ford) == "NVDA"


def test_scan_ignores_single_character_tickers(companies_with_ford):
    """The bug this replaces: the old substring matcher made Ford's "F" match
    every headline containing the letter f."""
    assert entities.resolve("Follow the cash flow", companies_with_ford) == []
    assert [m.ticker for m in entities.resolve("Ford reports earnings", companies_with_ford)] == ["F"]


def test_resolve_is_deterministic(companies):
    text = "Nvidia, AMD and TSMC all moved; Microsoft too"
    first = [m.ticker for m in entities.resolve(text, companies)]
    second = [m.ticker for m in entities.resolve(text, companies)]
    assert first == second
    assert first == ["NVDA", "AMD", "TSM", "MSFT"]
