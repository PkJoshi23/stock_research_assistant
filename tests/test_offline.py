"""The LLM-outage fallback: deterministic answers for known question shapes.

These tests are the difference between "the demo dies with the API" and "the
demo degrades". They run against the seeded graph fixture, with no model and no
network anywhere in the path.
"""

from __future__ import annotations

import pytest

import src.agent.master_agent as master
from src import relate_db
from src.agent import offline
from src.agent.master_agent import StockResearchAgent
from src.schema import PERSON, REL_WORKS_FOR, STOCK, SUBSIDIARY


@pytest.fixture
def agent(db_path, seed_graph):
    """Seed a store, close the seeding session, then hand over an agent."""
    seeding = relate_db.GraphSession()
    seed_graph(seeding.run)
    seeding.close()
    instance = StockResearchAgent()
    yield instance
    instance.close()


def _route_graph_query(monkeypatch, keys=None):
    monkeypatch.setattr(
        master,
        "detect_intent_and_extract_schema",
        lambda question: {
            "intent": "graph_query",
            "relevant_schema_keys": keys or ["corporate_structure"],
            "tickers": ["NVDA"],
            "companies": [],
            "reasoning": "stub",
        },
    )

# The seeded graph (tests/conftest.py) has: NVDA, AMD, TSM, MSFT in Technology,
# one CEO, three NVDA headlines, TSM->NVDA and NVDA->MSFT supply edges, and an
# NVDA-AMD competitor edge.


def _resolve_nvda(_question: str) -> str:
    return "NVDA"


@pytest.mark.parametrize(
    "question,shape",
    [
        ("Who supplies Nvidia?", "suppliers"),
        ("What subsidiaries does Nvidia own?", "subsidiaries"),
        ("Who is on the board of directors?", "board"),
        ("Which companies compete with Nvidia?", "competitors"),
        ("Any recent news about Nvidia?", "news"),
        ("What sector is Nvidia in?", "sector"),
        ("Tell me about the weather in Tokyo", None),
    ],
)
def test_classification(question: str, shape: str | None):
    assert offline.classify(question) == shape


def test_suppliers_answer_lists_tracked_supplier(seeded):
    result = offline.answer_without_llm(seeded, "Who supplies Nvidia?", _resolve_nvda)
    assert result is not None
    assert "TSM" in result["answer"]
    assert result["rows"]


def test_competitors_answer(seeded):
    result = offline.answer_without_llm(
        seeded, "Which companies compete with Nvidia?", _resolve_nvda
    )
    assert "AMD" in result["answer"]


def test_sector_answer(seeded):
    result = offline.answer_without_llm(seeded, "What sector is Nvidia in?", _resolve_nvda)
    assert "Technology" in result["answer"]


def test_news_answer_is_bounded(seeded):
    result = offline.answer_without_llm(
        seeded, "Any recent news about Nvidia?", _resolve_nvda
    )
    assert len(result["rows"]) <= 5
    assert "Nvidia" in result["answer"] or "NVDA" in result["answer"]


def test_subsidiaries_answer_when_present(seeded):
    seeded.run(
        f"MERGE (sub:{SUBSIDIARY} {{name: 'Mellanox'}}) "
        f"WITH sub MATCH (s:{STOCK} {{ticker: 'NVDA'}}) MERGE (s)-[:OWNS]->(sub)"
    )
    result = offline.answer_without_llm(
        seeded, "What subsidiaries does Nvidia own?", _resolve_nvda
    )
    assert "Mellanox" in result["answer"]


def test_board_returns_none_when_nobody_is_a_director(seeded):
    """The seeded person is a CEO, so there is no board to report."""
    assert (
        offline.answer_without_llm(
            seeded, "Who is on the board of directors?", _resolve_nvda
        )
        is None
    )


def test_board_answer_when_a_director_exists(seeded):
    seeded.run(
        f"MERGE (p:{PERSON} {{name: 'Ada Director'}}) SET p.role = 'Director' "
        f"WITH p MATCH (s:{STOCK} {{ticker: 'NVDA'}}) MERGE (p)-[:{REL_WORKS_FOR}]->(s)"
    )
    result = offline.answer_without_llm(
        seeded, "Who is on the board of directors?", _resolve_nvda
    )
    assert "Ada Director" in result["answer"]


def test_unknown_shape_returns_none(seeded):
    assert offline.answer_without_llm(seeded, "Tell me about Tokyo", _resolve_nvda) is None


def test_unresolvable_ticker_returns_none(seeded):
    assert (
        offline.answer_without_llm(seeded, "Who supplies them?", lambda _q: None) is None
    )


def test_empty_result_returns_none_rather_than_a_misleading_answer(seeded):
    """No competitors for a company with no edges -> let the caller report it."""
    assert (
        offline.answer_without_llm(
            seeded, "Which companies compete with Microsoft?", lambda _q: "MSFT"
        )
        is None
    )


def test_every_answer_is_labelled_degraded(seeded):
    result = offline.answer_without_llm(seeded, "Who supplies Nvidia?", _resolve_nvda)
    assert offline.DEGRADED_NOTICE in result["answer"]


# -- integration through the agent -------------------------------------------


def test_agent_falls_back_when_generation_fails(agent, monkeypatch):
    _route_graph_query(monkeypatch)
    monkeypatch.setattr(
        master,
        "generate_cypher_query",
        lambda *a, **k: {"action": "error", "cypher": None, "reasoning": "API down"},
    )

    result = agent.ask("Who supplies Nvidia?")

    assert result["degraded"] is True
    assert result["error"] == ""
    assert any(row.get("supplier") == "TSM" for row in result["rows"])
    assert "fixed graph queries" in result["answer"]


def test_agent_renders_rows_when_only_the_summariser_fails(agent, monkeypatch):
    _route_graph_query(monkeypatch)
    monkeypatch.setattr(
        master,
        "generate_cypher_query",
        lambda *a, **k: {
            "action": "graph_query",
            "cypher": "MATCH (sup)-[:SUPPLIES_TO]->(s:Stock {ticker: 'NVDA'}) "
                      "RETURN coalesce(sup.ticker, sup.name) AS supplier",
            "reasoning": "",
        },
    )

    def summariser_down(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(master.llm, "chat", summariser_down)

    result = agent.ask("Who supplies Nvidia?")

    assert result["degraded"] is True
    assert result["rows"]
    assert result["cypher"]  # the successful query is still shown


def test_unknown_question_still_reports_the_original_error(agent, monkeypatch):
    _route_graph_query(monkeypatch, keys=[])
    monkeypatch.setattr(
        master,
        "generate_cypher_query",
        lambda *a, **k: {"action": "error", "cypher": None, "reasoning": "API down"},
    )

    result = agent.ask("Tell me about the weather")

    assert result["degraded"] is False
    assert result["error"] == "API down"
