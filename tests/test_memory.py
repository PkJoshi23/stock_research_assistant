"""Conversation memory stored in the graph.

Two properties matter: the same question written two ways is one memory, and a
memory failure can never take down an answer.
"""

from __future__ import annotations

import pytest

from src import memory
from src.schema import QUERY, REL_ABOUT


def test_query_id_normalises_case_and_punctuation():
    assert memory.query_id("Who supplies Nvidia?") == memory.query_id(
        "  who supplies nvidia  "
    )
    assert memory.query_id("Who supplies Nvidia?") != memory.query_id(
        "Who competes with Nvidia?"
    )


def test_remember_creates_a_query_node(seeded):
    memory.remember(seeded, "Who supplies Nvidia?", "graph_query", ["NVDA"])
    nodes = seeded.run(f"MATCH (q:{QUERY}) RETURN q.text AS text, q.count AS count")
    assert len(nodes) == 1
    assert nodes[0]["text"] == "Who supplies Nvidia?"
    assert nodes[0]["count"] == 1


def test_remember_counts_repeats(seeded):
    first = memory.remember(seeded, "Who supplies Nvidia?", "graph_query", ["NVDA"])
    second = memory.remember(seeded, "who supplies nvidia?", "graph_query", ["NVDA"])

    assert first["asked_before"] == 0
    assert second["asked_before"] == 1
    count = seeded.run(f"MATCH (q:{QUERY}) RETURN q.count AS count")[0]["count"]
    assert count == 2


def test_remember_links_the_companies_asked_about(seeded):
    memory.remember(seeded, "Who supplies Nvidia?", "graph_query", ["NVDA"])
    links = seeded.run(
        f"MATCH (q:{QUERY})-[:{REL_ABOUT}]->(s:Stock) RETURN s.ticker AS ticker"
    )
    assert [row["ticker"] for row in links] == ["NVDA"]


def test_unknown_ticker_still_records_the_question(seeded):
    info = memory.remember(seeded, "Tell me about ZZZZ", "graph_query", ["ZZZZ"])
    assert info["asked_before"] == 0
    assert seeded.run(f"MATCH (q:{QUERY}) RETURN count(q) AS n")[0]["n"] == 1
    assert seeded.run(f"MATCH ()-[r:{REL_ABOUT}]->() RETURN count(r) AS n")[0]["n"] == 0


def test_recent_returns_newest_first(seeded):
    memory.remember(seeded, "first question", "graph_query", [])
    memory.remember(seeded, "second question", "graph_query", [])
    texts = [row["text"] for row in memory.recent(seeded, limit=5)]
    assert texts[0] == "second question"


def test_interest_profile_counts_questions_per_company(seeded):
    memory.remember(seeded, "Who supplies Nvidia?", "graph_query", ["NVDA"])
    memory.remember(seeded, "Nvidia competitors?", "graph_query", ["NVDA"])
    memory.remember(seeded, "AMD suppliers?", "graph_query", ["AMD"])

    profile = memory.interest_profile(seeded, limit=3)
    assert profile[0]["ticker"] == "NVDA"
    assert profile[0]["questions"] == 2


def test_ticker_history(seeded):
    memory.remember(seeded, "Who supplies Nvidia?", "graph_query", ["NVDA"])
    history = memory.ticker_history(seeded, "NVDA")
    assert history and history[0]["text"] == "Who supplies Nvidia?"


def test_memory_failure_never_raises(seeded, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("graph unavailable")

    monkeypatch.setattr(seeded, "run", boom)

    info = memory.remember(seeded, "Who supplies Nvidia?", "graph_query", ["NVDA"])
    assert info["asked_before"] == 0
    assert memory.recent(seeded) == []
    assert memory.interest_profile(seeded) == []
