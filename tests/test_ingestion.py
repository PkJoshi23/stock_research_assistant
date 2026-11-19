"""Ingestion: idempotency, de-duplication and the edge cases that bit us.

Two behaviours are load-bearing for the demo:
  * re-running ingestion must not duplicate nodes or edges (it is run repeatedly
    while tuning the curated relationship list);
  * re-running the SEC loader must not duplicate people, subsidiaries or
    supplier edges.
"""

from __future__ import annotations

import pytest

from src import config, relate_db
from src.ingestion import ingest_stock_data as stock_ing

SNAPSHOT = {
    "ticker": "NVDA",
    "name": "NVIDIA Corporation",
    "sector": "Technology",
    "industry": "Semiconductors",
    "market_cap": 3.4e12,
    "pe_ratio": 55.2,
    "current_price": 150.0,
    "aliases": ["Nvidia"],
}


# -- stock ingestion ---------------------------------------------------------


def test_upsert_stock_creates_node_sector_and_edge(conn):
    stock_ing.upsert_stock(conn, SNAPSHOT)
    assert relate_db.counts(conn) == {"Stock": 1, "Sector": 1}
    edges = relate_db.run(conn, "MATCH ()-[r:IN_SECTOR]->() RETURN count(r) AS n")
    assert edges[0]["n"] == 1


def test_upsert_stock_is_idempotent_and_updates_values(conn):
    stock_ing.upsert_stock(conn, SNAPSHOT)
    stock_ing.upsert_stock(conn, {**SNAPSHOT, "current_price": 199.0})
    assert relate_db.counts(conn) == {"Stock": 1, "Sector": 1}
    assert relate_db.run(conn, "MATCH ()-[r:IN_SECTOR]->() RETURN count(r) AS n")[0]["n"] == 1
    price = relate_db.run(conn, "MATCH (s:Stock {ticker: 'NVDA'}) RETURN s.current_price AS p")
    assert price[0]["p"] == 199.0


def test_relationship_write_is_idempotent(conn):
    stock_ing.upsert_stock(conn, SNAPSHOT)
    stock_ing.upsert_stock(conn, {**SNAPSHOT, "ticker": "AMD", "name": "AMD", "aliases": []})
    for _ in range(3):
        stock_ing.write_relationship(conn, "NVDA", "AMD", "COMPETES_WITH")
    assert relate_db.run(conn, "MATCH ()-[r:COMPETES_WITH]->() RETURN count(r) AS n")[0]["n"] == 1


def test_person_write_is_idempotent(conn):
    stock_ing.upsert_stock(conn, SNAPSHOT)
    for _ in range(2):
        stock_ing.write_person(conn, "Jensen Huang", "CEO", "NVDA")
    assert relate_db.counts(conn).get("Person") == 1
    assert relate_db.run(conn, "MATCH ()-[r:WORKS_FOR]->() RETURN count(r) AS n")[0]["n"] == 1


def test_aliases_are_written(conn):
    stock_ing.upsert_stock(conn, SNAPSHOT)
    rows = relate_db.run(conn, "MATCH (s:Stock {ticker: 'NVDA'}) RETURN s.aliases AS a")
    assert rows[0]["a"] == ["Nvidia"]

