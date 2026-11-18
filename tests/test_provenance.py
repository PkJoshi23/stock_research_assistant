"""Provenance: every answer should be auditable from the graph it came from."""

from __future__ import annotations

import pytest

from src import provenance
from src.schema import PERSON, REL_SUPPLIES_TO, REL_WORKS_FOR, STOCK


def _types(sources: list[dict]) -> list[str]:
    return [source["type"] for source in sources]


def test_rows_produce_a_graph_source(seeded):
    sources = provenance.collect(seeded, "NVDA", [{"ticker": "NVDA"}])
    assert _types(sources)[0] == "graph"
    assert "1 row" in sources[0]["label"]


def test_no_rows_means_no_graph_source(seeded):
    sources = provenance.collect(seeded, "NVDA", [])
    assert "graph" not in _types(sources)


def test_filing_evidence_is_reported_when_edges_carry_it(seeded):
    seeded.run(
        f"""
        MATCH (p:{PERSON} {{name: 'Jensen Huang'}}), (s:{STOCK} {{ticker: 'NVDA'}})
        MERGE (p)-[r:{REL_WORKS_FOR}]->(s) SET r.source = 'DEF 14A'
        """
    )
    seeded.run(
        f"""
        MATCH (a:{STOCK} {{ticker: 'TSM'}})-[r:{REL_SUPPLIES_TO}]->(b:{STOCK} {{ticker: 'NVDA'}})
        SET r.source = '10-K'
        """
    )

    sources = provenance.collect(seeded, "NVDA", [{"x": 1}])
    labels = " ".join(source["label"] for source in sources)

    assert "DEF 14A" in labels
    assert "10-K" in labels


def test_headline_urls_are_surfaced_and_capped(seeded):
    rows = [
        {"title": f"Headline {index}", "url": f"https://ex.com/{index}"}
        for index in range(5)
    ]
    sources = provenance.collect(seeded, None, rows)
    news = [source for source in sources if source["type"] == "news"]
    assert len(news) == provenance._MAX_HEADLINES
    assert news[0]["url"] == "https://ex.com/0"


def test_rows_without_urls_do_not_become_news_sources(seeded):
    sources = provenance.collect(seeded, None, [{"title": "no link here"}])
    assert "news" not in _types(sources)


def test_snapshot_timestamp_is_reported_when_present(seeded):
    seeded.run(
        f"MATCH (s:{STOCK} {{ticker: 'NVDA'}}) SET s.updated_at = $now",
        {"now": __import__("datetime").datetime(2026, 11, 18, 9, 0)},
    )
    sources = provenance.collect(seeded, "NVDA", [{"x": 1}])
    snapshot = [source for source in sources if source["type"] == "snapshot"]
    assert snapshot and "2026-11-18" in snapshot[0]["label"]


def test_collect_is_safe_without_a_ticker(seeded):
    assert provenance.collect(seeded, None, []) == []


def test_live_quote_source_states_it_was_not_stored():
    sources = provenance.live_quote_source("NVDA", 153.5, "USD", "2026-11-18T10:30:00")
    assert sources[0]["type"] == "live_quote"
    assert "not persisted" in sources[0]["detail"]
    assert "153.50" in sources[0]["label"]
