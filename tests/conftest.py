"""Shared fixtures.

Rules for this suite:

* **Never touch the real store.** ``config.DB_PATH`` is redirected to ``tmp_path``
  before anything connects, so a test run cannot corrupt ``data/stock_graph.rl``
  (and cannot hit Relate's single-writer lease if the Streamlit app is open).
* **Never call the network or an LLM.** Anything that would do so is
  monkeypatched in the test that needs it.
* **Seed through the public API.** Graph fixtures are built with the same
  parameterised Cypher the ingestion scripts use, so the tests exercise the real
  write path rather than a hand-rolled shortcut.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, relate_db  # noqa: E402


@pytest.fixture
def db_path(tmp_path, monkeypatch) -> Path:
    """Point the whole application at an isolated store."""
    path = tmp_path / "test_graph.rl"
    monkeypatch.setattr(config, "DB_PATH", path)
    return path


@pytest.fixture
def conn(db_path):
    """Raw connection for single-threaded ingestion-style tests."""
    connection = relate_db.connect()
    relate_db.ensure_schema(connection)
    yield connection
    connection.close()


@pytest.fixture
def session(db_path):
    """Thread-safe session, the way the agent uses the store."""
    graph = relate_db.GraphSession()
    yield graph
    graph.close()


def seed_demo_graph(run) -> None:
    """Populate a small but structurally complete graph.

    ``run`` is a ``(cypher, params) -> rows`` callable, so this works with both a
    raw connection (``partial(relate_db.run, conn)``) and a ``GraphSession``.
    """
    stocks = [
        ("NVDA", "NVIDIA Corporation", "Technology", ["Nvidia"]),
        ("AMD", "Advanced Micro Devices, Inc.", "Technology", ["AMD"]),
        ("TSM", "Taiwan Semiconductor Manufacturing Company Limited", "Technology", ["TSMC"]),
        ("MSFT", "Microsoft Corporation", "Technology", ["Microsoft"]),
    ]
    for ticker, name, sector, aliases in stocks:
        run(
            "MERGE (s:Stock {ticker: $ticker}) "
            "SET s.name = $name, s.sector = $sector, s.aliases = $aliases, "
            "    s.current_price = 100.0, s.market_cap = 1000000000.0, s.pe_ratio = 20.0",
            {"ticker": ticker, "name": name, "sector": sector, "aliases": aliases},
        )
        run("MERGE (sec:Sector {name: $name})", {"name": sector})
        run(
            "MATCH (s:Stock {ticker: $ticker}), (sec:Sector {name: $sector}) "
            "MERGE (s)-[:IN_SECTOR]->(sec)",
            {"ticker": ticker, "sector": sector},
        )

    run(
        "MATCH (a:Stock {ticker: 'NVDA'}), (b:Stock {ticker: 'AMD'}) "
        "MERGE (a)-[:COMPETES_WITH]->(b)"
    )
    run(
        "MATCH (a:Stock {ticker: 'TSM'}), (b:Stock {ticker: 'NVDA'}) "
        "MERGE (a)-[:SUPPLIES_TO]->(b)"
    )
    run(
        "MATCH (a:Stock {ticker: 'NVDA'}), (b:Stock {ticker: 'MSFT'}) "
        "MERGE (a)-[:SUPPLIES_TO]->(b)"
    )
    run(
        "MERGE (p:Person {name: 'Jensen Huang'}) SET p.role = 'CEO' "
        "WITH p MATCH (s:Stock {ticker: 'NVDA'}) MERGE (p)-[:WORKS_FOR]->(s)"
    )

    # Three NVDA headlines (so the per-stock cap can be exercised) and one AMD.
    news = [
        ("https://ex.com/a", "Nvidia beats earnings", "NVDA", "Reuters"),
        ("https://ex.com/b", "Nvidia supply chain tightens", "NVDA", "Yahoo Finance"),
        ("https://ex.com/c", "Nvidia shares fall", "NVDA", "Trefis"),
        ("https://ex.com/d", "AMD launches new chip", "AMD", "Reuters"),
    ]
    for url, title, ticker, source in news:
        run(
            "MERGE (n:NewsArticle {url: $url}) SET n.title = $title, n.source = $source, "
            "    n.summary = 'stub', n.embedding = $embedding",
            {"url": url, "title": title, "source": source, "embedding": [0.01] * config.EMBEDDING_DIM},
        )
        run(
            "MATCH (s:Stock {ticker: $ticker}), (n:NewsArticle {url: $url}) "
            "MERGE (s)-[:MENTIONED_IN]->(n)",
            {"ticker": ticker, "url": url},
        )


@pytest.fixture
def seeded(session):
    """A ``GraphSession`` backed by the demo graph."""
    seed_demo_graph(session.run)
    return session


@pytest.fixture
def seed_graph():
    """The seeding helper itself, for tests that must control when it runs.

    The agent fixture has to seed, close, and only then construct the agent:
    Relate grants one writable lease per store, so seeding while the agent holds
    the connection is refused.
    """
    return seed_demo_graph
