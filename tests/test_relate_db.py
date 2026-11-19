"""Data-layer behaviour: rows, indexes, and the thread-safety guarantee.

The thread test is the important one. Relate connections are thread-affine
(``threadsafety == 1``) and Streamlit runs every rerun on a new thread; caching a
raw connection produced ``ProgrammingError`` on every question after the first.
``GraphSession`` exists to fix that, so it gets an explicit regression test.
"""

from __future__ import annotations

import threading

import pytest

from src import relate_db
from src.schema import STOCK


def test_run_returns_plain_dict_rows(conn):
    relate_db.run(conn, "CREATE (:Stock {ticker: $t, name: $n})", {"t": "NVDA", "n": "NVIDIA"})
    rows = relate_db.run(conn, "MATCH (s:Stock) RETURN s.ticker AS ticker, s.name AS name")
    assert rows == [{"ticker": "NVDA", "name": "NVIDIA"}]
    assert type(rows[0]) is dict


def test_ensure_schema_creates_the_ticker_lookup_index(conn):
    assert "stock_ticker_idx" in relate_db.index_names(conn)


def test_ensure_schema_is_idempotent(conn):
    relate_db.ensure_schema(conn)
    relate_db.ensure_schema(conn)
    assert relate_db.index_names(conn).count("stock_ticker_idx") == 1


def test_counts_groups_by_label(conn):
    relate_db.run(conn, "CREATE (:Stock {ticker: 'A'})")
    relate_db.run(conn, "CREATE (:Sector {name: 'Tech'})")
    assert relate_db.counts(conn) == {STOCK: 1, "Sector": 1}


def test_wipe_removes_nodes_but_keeps_indexes(conn):
    relate_db.run(conn, "CREATE (:Stock {ticker: 'A'})")
    relate_db.wipe(conn)
    assert relate_db.counts(conn) == {}
    assert "stock_ticker_idx" in relate_db.index_names(conn)


def test_session_works_from_a_different_thread(seeded):
    """The regression: a raw connection raises here, a session must not."""
    result: dict = {}

    def query_from_another_thread() -> None:
        try:
            result["rows"] = seeded.run(
                f"MATCH (s:{STOCK}) RETURN count(s) AS n"
            )
        except Exception as exc:  # noqa: BLE001 - the point of the test
            result["error"] = exc

    thread = threading.Thread(target=query_from_another_thread)
    thread.start()
    thread.join(timeout=30)

    assert "error" not in result, f"session not thread-safe: {result.get('error')}"
    assert result["rows"] == [{"n": 4}]


def test_session_close_is_idempotent(db_path):
    graph = relate_db.GraphSession()
    graph.close()
    graph.close()  # must not raise


def test_raw_connection_is_thread_affine(conn):
    """Documents the constraint the session wrapper exists to work around."""
    error: dict = {}

    def use_connection_elsewhere() -> None:
        try:
            conn.execute("RETURN 1 AS x")
        except Exception as exc:  # noqa: BLE001
            error["type"] = type(exc).__name__

    thread = threading.Thread(target=use_connection_elsewhere)
    thread.start()
    thread.join(timeout=30)

    assert error.get("type") == "ProgrammingError"
