"""Step 1 + 2: intent classification and schema-key extraction.

The fallback path matters more than the happy path: if DeepSeek is rate-limited
during the demo, the agent must still deflect historical questions and never
answer them from a graph that has no price history.
"""

from __future__ import annotations

import pytest

from src import llm
from src.agent import agent_router as router


@pytest.fixture(autouse=True)
def _offline_llm(monkeypatch):
    """Default every test to 'LLM is down' unless it replaces this."""

    def boom(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(llm, "chat_json", boom)


@pytest.mark.parametrize(
    "question",
    [
        "How has Tesla's stock performed since 2020?",
        "Show me a 5-year chart of Apple",
        "What was Microsoft's price last month?",
        "How has NVDA trended over the past year?",
        "Give me the moving average of Intel",
        "What has Google's dividend history looked like?",
    ],
)
def test_historical_questions_are_deferred_without_the_llm(question: str):
    result = router.detect_intent_and_extract_schema(question)
    assert result["intent"] == router.INTENT_HISTORICAL
    assert result["relevant_schema_keys"] == []


@pytest.mark.parametrize(
    "question",
    [
        "What is NVDA trading at right now?",
        "current price of Microsoft please",
        "Is Apple worth buying today?",
    ],
)
def test_live_price_questions_are_classified_as_live(question: str):
    result = router.detect_intent_and_extract_schema(question)
    assert result["intent"] == router.INTENT_LIVE_PRICE


@pytest.mark.parametrize(
    "question",
    [
        "Who supplies Nvidia?",
        "Which companies compete with Microsoft?",
        "What recent news mentions Tesla?",
    ],
)
def test_relationship_questions_default_to_graph_queries(question: str):
    result = router.detect_intent_and_extract_schema(question)
    assert result["intent"] == router.INTENT_GRAPH_QUERY
    assert result["relevant_schema_keys"] == ["stock_snapshot"]


def test_historical_wins_over_graph_wording():
    """A past-tense performance question must not slip through as a graph query."""
    result = router.detect_intent_and_extract_schema(
        "Which sector was Nvidia in and how has it performed since 2021?"
    )
    assert result["intent"] == router.INTENT_HISTORICAL


def test_llm_json_is_validated_against_the_dictionary(monkeypatch):
    monkeypatch.setattr(
        llm,
        "chat_json",
        lambda **kwargs: {
            "intent": "graph_query",
            "relevant_schema_keys": ["competitors", "made_up_key"],
            "tickers": ["nvda", "TOOLONG", "AM D"],
            "companies": ["Nvidia"],
            "reasoning": "because",
        },
    )
    result = router.detect_intent_and_extract_schema("Who competes with Nvidia?")
    assert result["relevant_schema_keys"] == ["competitors"]
    assert result["tickers"] == ["NVDA"]
    assert result["companies"] == ["Nvidia"]


def test_unknown_intent_falls_back(monkeypatch):
    monkeypatch.setattr(
        llm, "chat_json", lambda **kwargs: {"intent": "make_me_a_sandwich"}
    )
    result = router.detect_intent_and_extract_schema("Who supplies Nvidia?")
    assert result["intent"] == router.INTENT_GRAPH_QUERY


def test_graph_query_without_keys_gets_the_snapshot_key(monkeypatch):
    monkeypatch.setattr(
        llm,
        "chat_json",
        lambda **kwargs: {"intent": "graph_query", "relevant_schema_keys": []},
    )
    result = router.detect_intent_and_extract_schema("Who supplies Nvidia?")
    assert result["relevant_schema_keys"] == ["stock_snapshot"]


def test_non_json_response_does_not_raise(monkeypatch):
    def bad_json(**kwargs):
        raise ValueError("not json")

    monkeypatch.setattr(llm, "chat_json", bad_json)
    result = router.detect_intent_and_extract_schema("Who supplies Nvidia?")
    assert result["intent"] in router.VALID_INTENTS


@pytest.mark.parametrize(
    "raw,expected",
    [
        (["aapl", "MSFT"], ["AAPL", "MSFT"]),
        (["TOOLONG", "AM D", ""], []),
        (["NVDA", "NVDA"], ["NVDA"]),
        (["A", "B", "C", "D", "E", "F", "G"], ["A", "B", "C", "D", "E"]),
        (None, []),
        ("NVDA", []),
    ],
)
def test_ticker_cleanup(raw, expected):
    assert router._clean_tickers(raw) == expected
