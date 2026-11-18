"""Orchestration: intent dispatch, the deflection branch, and ticker resolution.

Nothing here calls DeepSeek or Yahoo: the router, the generator, the summariser
and yfinance are all replaced. What is *not* replaced is the graph itself, so the
Cypher path runs against a real (temporary) Relate store.
"""

from __future__ import annotations

import pytest

import src.agent.master_agent as master
from src import quotes, relate_db
from src.agent.master_agent import HISTORICAL_MESSAGE, StockResearchAgent


@pytest.fixture
def agent(db_path, seed_graph):
    """Seed a store, close the seeding session, then hand over an agent.

    The seed session must be closed first: Relate allows one writable open per
    store, so an agent created alongside it would be refused the lease.
    """
    seeding = relate_db.GraphSession()
    seed_graph(seeding.run)
    seeding.close()

    instance = StockResearchAgent()
    yield instance
    instance.close()


def route(monkeypatch, intent: str, keys=None, tickers=None):
    monkeypatch.setattr(
        master,
        "detect_intent_and_extract_schema",
        lambda question: {
            "intent": intent,
            "relevant_schema_keys": keys or [],
            "tickers": tickers or [],
            "companies": [],
            "reasoning": "stubbed",
        },
    )


# -- historical --------------------------------------------------------------


def test_historical_question_is_deferred_together(agent, monkeypatch):
    route(monkeypatch, master.INTENT_HISTORICAL)

    def explode(*args, **kwargs):
        raise AssertionError("no LLM or graph access for a historical question")

    monkeypatch.setattr(master.llm, "chat", explode)
    monkeypatch.setattr(master, "generate_cypher_query", explode)

    result = agent.ask("How has Tesla done since 2020?")
    assert result["answer"] == HISTORICAL_MESSAGE
    assert result["cypher"] is None
    assert result["rows"] == []
    assert result["error"] == ""


# -- graph queries -----------------------------------------------------------


def test_graph_question_returns_rows_and_summary(agent, monkeypatch):
    route(monkeypatch, master.INTENT_GRAPH_QUERY, ["stock_snapshot"])
    monkeypatch.setattr(
        master,
        "generate_cypher_query",
        lambda *a, **k: {
            "action": "graph_query",
            "cypher": "MATCH (s:Stock {ticker: 'NVDA'}) RETURN s.ticker AS ticker",
            "reasoning": "",
        },
    )
    monkeypatch.setattr(master.llm, "chat", lambda *a, **k: "NVIDIA is tracked.")

    result = agent.ask("Tell me about NVDA")
    assert result["rows"] == [{"ticker": "NVDA"}]
    assert result["answer"] == "NVIDIA is tracked."
    assert "MATCH" in result["cypher"]
    assert result["error"] == ""


def test_empty_result_is_reported_without_an_llm_call(agent, monkeypatch):
    route(monkeypatch, master.INTENT_GRAPH_QUERY, ["stock_snapshot"])
    monkeypatch.setattr(
        master,
        "generate_cypher_query",
        lambda *a, **k: {
            "action": "graph_query",
            "cypher": "MATCH (s:Stock {ticker: 'ZZZZ'}) RETURN s.ticker AS ticker",
            "reasoning": "",
        },
    )
    monkeypatch.setattr(master.llm, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no LLM")))

    result = agent.ask("Tell me about a company that does not exist")
    assert result["rows"] == []
    assert "no data matching" in result["answer"]
    assert result["error"] == ""


def test_generation_failure_is_surfaced(agent, monkeypatch):
    route(monkeypatch, master.INTENT_GRAPH_QUERY, ["stock_snapshot"])
    monkeypatch.setattr(
        master,
        "generate_cypher_query",
        lambda *a, **k: {"action": "error", "cypher": None, "reasoning": "model said no"},
    )
    result = agent.ask("something weird")
    assert "couldn't turn that into a graph query" in result["answer"]
    assert result["error"] == "model said no"


def test_failed_execution_is_surfaced(agent, monkeypatch):
    route(monkeypatch, master.INTENT_GRAPH_QUERY, ["stock_snapshot"])
    monkeypatch.setattr(
        master,
        "generate_cypher_query",
        lambda *a, **k: {"action": "graph_query", "cypher": "MATCH (s:Stock RETURN", "reasoning": ""},
    )
    monkeypatch.setattr(master.llm, "chat", lambda *a, **k: "MATCH (s:Stock RETURN")
    result = agent.ask("broken query")
    assert result["error"]
    assert "couldn't run that query" in result["answer"]


# -- live price --------------------------------------------------------------


@pytest.fixture
def fake_quote(monkeypatch):
    """Replace the quote adapter so the live-price branch never touches the network.

    The agent no longer reaches into yfinance directly: both the live answer and
    ingestion go through ``quotes.fetch``, so this is the only seam a test needs.
    """
    monkeypatch.setattr(
        master.quotes,
        "fetch",
        lambda ticker: quotes.Quote(
            ticker=ticker,
            price=123.45,
            currency="USD",
            source="fast_info",
            at="2026-11-18T10:00:00",
        ),
    )


def test_live_price_is_enriched_with_graph_context(agent, monkeypatch, fake_quote):
    route(monkeypatch, master.INTENT_LIVE_PRICE, ["stock_snapshot"], ["NVDA"])
    result = agent.ask("What is NVDA trading at right now?")
    assert "$123.45" in result["answer"]
    assert "AMD" in result["answer"]  # competitor from the graph
    assert "TSM" in result["answer"]  # supplier from the graph


def test_live_price_is_not_persisted(agent, monkeypatch, fake_quote):
    route(monkeypatch, master.INTENT_LIVE_PRICE, ["stock_snapshot"], ["NVDA"])
    agent.ask("What is NVDA trading at right now?")
    stored = agent.db.run("MATCH (s:Stock {ticker: 'NVDA'}) RETURN s.current_price AS p")
    assert stored[0]["p"] == 100.0  # the seeded snapshot, untouched


def test_live_price_without_a_ticker_asks_for_one(agent, monkeypatch, fake_quote):
    route(monkeypatch, master.INTENT_LIVE_PRICE, ["stock_snapshot"], [])
    result = agent.ask("What is the price right now?")
    assert "couldn't tell which ticker" in result["answer"]
    assert result["error"] == ""


# -- robustness --------------------------------------------------------------


def test_routing_failure_never_raises(agent, monkeypatch):
    def boom(question):
        raise RuntimeError("router exploded")

    monkeypatch.setattr(master, "detect_intent_and_extract_schema", boom)
    result = agent.ask("anything")
    assert result["error"]
    assert "could not classify" in result["answer"]


def test_result_always_has_the_canonical_shape(agent, monkeypatch):
    route(monkeypatch, master.INTENT_HISTORICAL)
    result = agent.ask("history?")
    assert set(result) == {
        "intent",
        "schema_keys",
        "tickers",
        "cypher",
        "rows",
        "answer",
        "error",
        "sources",
        "usage",
        "memory",
        "degraded",
        # The coverage gate's decision, always present so the UI can show it.
        "gate",
        "memory",
        "degraded",
    }


def test_result_carries_usage_and_memory(agent, monkeypatch):
    route(monkeypatch, master.INTENT_HISTORICAL)
    result = agent.ask("history?")

    assert result["usage"]["calls"] == 0
    assert result["usage"]["wall_ms"] >= 0
    assert result["memory"]["asked_before"] == 0


def test_repeated_question_is_remembered(agent, monkeypatch):
    route(monkeypatch, master.INTENT_HISTORICAL)

    agent.ask("how has nvda done since 2020?")
    second = agent.ask("How has NVDA done since 2020?")

    assert second["memory"]["asked_before"] == 1


# -- ticker resolution -------------------------------------------------------


def test_alias_resolves_to_ticker(agent):
    assert agent._resolve_ticker("what about TSMC then", []) == "TSM"
    assert agent._resolve_ticker("Who runs Microsoft?", []) == "MSFT"


def test_unknown_ticker_is_not_guessed(agent):
    assert agent._resolve_ticker("what about ZZZZ", []) is None


def test_tickers_from_accepts_a_question_or_a_list(agent):
    assert agent._tickers_from(["nvda"]) == ["NVDA"]
    assert agent._tickers_from("Who supplies Nvidia?") == ["NVDA"]


# -- visualization hand-off --------------------------------------------------


def test_visualization_returns_html_for_known_entities(agent):
    html = agent.generate_graph_visualization(["NVDA"])
    assert "vis.DataSet" in html


def test_visualization_returns_empty_string_when_there_is_nothing_to_draw(agent):
    assert agent.generate_graph_visualization([]) == ""
    assert agent.generate_graph_visualization(["NOPE"]) == ""
