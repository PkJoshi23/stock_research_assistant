"""The deterministic coverage gate and its agent wiring.

The point of the gate is economic: an untracked ticker must be answered without
calling the model at all. These tests assert exactly that (a monkeypatched model
that fails if touched), not just that the wording is right.
"""

from __future__ import annotations

import pytest

import src.llm as llm
from src import coverage, quotes, relate_db
from src.agent.master_agent import StockResearchAgent

REGISTRY = [
    {"ticker": "AAOI", "title": "APPLIED OPTOELECTRONICS, INC.", "cik": "0001158114"},
    {"ticker": "NVDA", "title": "NVIDIA CORP", "cik": "0001045810"},
    {"ticker": "ON", "title": "ON SEMICONDUCTOR CORP", "cik": "0001097864"},
    {"ticker": "AI", "title": "C3.ai, Inc.", "cik": "0001577526"},
    {"ticker": "OPEN", "title": "Opendoor Technologies Inc.", "cik": "0001801169"},
]

COMPANIES = [
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "aliases": []},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "aliases": []},
]


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(coverage.ticker_directory, "load_directory", lambda **kwargs: REGISTRY)


# -- the gate ----------------------------------------------------------------


def test_unknown_uppercase_ticker_is_blocked(registry):
    gate = coverage.assess("who supplies to AAOI?", COMPANIES)

    assert gate.blocked
    assert gate.untracked == [("AAOI", "APPLIED OPTOELECTRONICS, INC.")]


def test_unknown_lowercase_ticker_is_detected(registry):
    """Users type "aaoi", not "AAOI"; four characters is specific enough."""
    assert coverage.assess("who supplies to aaoi?", COMPANIES).blocked


def test_short_or_ambiguous_tokens_do_not_block(registry):
    for question in (
        "who supplies on the market?",
        "what does ai mean here?",
        "open the market data about chips",
    ):
        assert not coverage.assess(question, COMPANIES).blocked, question


def test_a_tracked_mention_disables_the_gate(registry):
    gate = coverage.assess("compare NVDA and AAOI", COMPANIES)

    assert gate.tracked == ["NVDA"]
    assert not gate.blocked


def test_thematic_question_passes_through(registry):
    gate = coverage.assess("what does the market say about AI chip demand?", COMPANIES)

    assert gate.untracked == []
    assert not gate.blocked


def test_gate_works_without_a_cache_or_sec_contact(tmp_path, monkeypatch):
    """The deployed app has neither a .cache nor SEC_CONTACT_EMAIL.

    Relying on the downloaded registry made the gate a silent no-op on Cloud,
    so the app kept running the full pipeline. The bundled registry must be
    enough.
    """
    from src import config

    monkeypatch.setattr(config, "SEC_CACHE_DIR", tmp_path)  # empty: no cache
    monkeypatch.setattr(config, "SEC_CONTACT_EMAIL", "")

    gate = coverage.assess("who supplies to aaoi?", COMPANIES)

    assert gate.blocked
    assert gate.untracked[0][0] == "AAOI"


def test_price_question_detection():
    assert coverage.looks_like_price_question("what is AAOI trading at?")
    assert coverage.looks_like_price_question("AAOI price?")
    assert not coverage.looks_like_price_question("who supplies AAOI?")


def test_refusal_names_the_company_and_the_coverage(registry):
    gate = coverage.assess("who supplies AAOI?", COMPANIES)

    text = coverage.refusal_text(gate, ["MSFT", "NVDA"])

    assert "AAOI" in text and "Applied Optoelectronics" in text
    assert "MSFT" in text and "NVDA" in text


# -- agent wiring ------------------------------------------------------------


@pytest.fixture
def agent(db_path, seed_graph):
    seeding = relate_db.GraphSession()
    seed_graph(seeding.run)
    seeding.close()
    instance = StockResearchAgent()
    yield instance
    instance.close()


@pytest.fixture
def no_model(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("the coverage gate must not call the model")

    monkeypatch.setattr(llm, "chat", _forbidden)
    monkeypatch.setattr(llm, "chat_json", _forbidden)


def test_unknown_ticker_is_answered_with_zero_model_calls(agent, registry, no_model):
    result = agent.ask("who supplies to aaoi?")

    assert result["intent"] == "coverage_miss"
    assert "AAOI" in result["answer"]
    assert result["coverage"]["can_fetch"] is True
    assert result["usage"]["calls"] == 0


def test_unknown_ticker_price_is_answered_live(agent, registry, no_model, monkeypatch):
    monkeypatch.setattr(
        quotes,
        "fetch",
        lambda ticker: quotes.Quote(
            ticker=ticker, price=12.5, currency="USD", source="yfinance", at="2026-11-19"
        ),
    )

    result = agent.ask("what is AAOI trading at?")

    assert result["intent"] == "live_price_fetch"
    assert "12.50" in result["answer"]
    assert result["usage"]["calls"] == 0


def test_tracked_but_empty_result_offers_the_fetch(agent, registry, monkeypatch):
    """'suppliers of CRWV?' - tracked company, no matching edges.

    The node exists, so the coverage gate passes; without this the UI dead-ends
    on a canned "no data" message and never offers the SEC fetch.
    """
    import src.agent.master_agent as master

    monkeypatch.setattr(
        master,
        "detect_intent_and_extract_schema",
        lambda question: {
            "intent": "graph_query",
            "relevant_schema_keys": ["supply_chain"],
            "tickers": ["NVDA"],
            "reasoning": "",
        },
    )
    monkeypatch.setattr(
        master,
        "generate_cypher_query",
        lambda *a, **k: {"action": "graph_query", "cypher": "MATCH (n) RETURN n", "reasoning": ""},
    )
    monkeypatch.setattr(
        master,
        "execute_cypher_with_retry",
        lambda *a, **k: {"rows": [], "cypher": "MATCH (n) RETURN n", "error": "", "attempts": 1},
    )

    result = agent.ask("who supplies nvidia?")

    assert result["rows"] == []
    assert result["enrich"]["ticker"] == "NVDA"
    assert "no matching edges" in result["answer"]


def test_tracked_ticker_still_runs_the_pipeline(agent, registry, monkeypatch):
    """The gate must not swallow questions the graph can answer."""
    import src.agent.master_agent as master

    called: list[str] = []

    def fake_route(question):
        called.append(question)
        return {
            "intent": "graph_query",
            "relevant_schema_keys": ["relationships"],
            "tickers": ["NVDA"],
            "reasoning": "",
        }

    monkeypatch.setattr(master, "detect_intent_and_extract_schema", fake_route)
    monkeypatch.setattr(
        master,
        "generate_cypher_query",
        lambda *a, **k: {"action": "graph_query", "cypher": "RETURN 1 AS x", "reasoning": ""},
    )
    # Empty rows keep this test model-free: the no-data branch is deterministic.
    monkeypatch.setattr(
        master,
        "execute_cypher_with_retry",
        lambda *a, **k: {"rows": [], "cypher": "RETURN 1 AS x", "error": "", "attempts": 1},
    )

    result = agent.ask("who supplies nvidia?")

    assert called, "a tracked ticker must reach the router"
    assert result["intent"] == "graph_query"
    assert "coverage" not in result
