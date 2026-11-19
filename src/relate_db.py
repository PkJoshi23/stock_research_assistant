"""The only place that opens and configures the Relate store.

Every module used to open its own connection with a different path
(``Relate("stock_graph.db")``, ``Relate("data/stock_graph.db")``, ...), which is
how you end up demoing against an empty database. All access goes through here.

Relate facts this module encodes (verified against 1.0.2):
  * the constructor is ``relate.connect(path)``, there is no ``Relate`` class;
  * the connection method is ``execute(cypher, params)``, there is no ``query``;
  * ``execute`` returns a Cursor, so rows must be read with ``fetchall()`` /
    ``data()`` / ``single()``;
  * ``cursor.data()`` turns nodes/relationships/paths inside a row into plain
    dicts, which is what Streamlit and ``json.dumps`` need.
"""

from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import relate

from . import config
from .schema import STOCK


def connect(*, read_only: bool = False) -> relate.Connection:
    """Open the project store, creating the ``data/`` directory if needed.

    Relate allows exactly one writable open per store (single-writer lease).
    The Streamlit app keeps one cached connection; ingestion scripts open their
    own and must not run while the app holds the lease.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return relate.connect(str(config.DB_PATH), read_only=read_only)


def run(
    conn: relate.Connection, cypher: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Execute a statement and return plain rows.

    ``cursor.data()`` is preferred over ``fetchall()`` because it recursively
    converts graph entities; a raw ``Node`` in a row is not JSON serialisable
    and renders as an opaque object in Streamlit.
    """
    cursor = conn.execute(cypher, params or {})
    try:
        # ``Record`` is a dict subclass; converting keeps the promise in this
        # function's name literal (exactly ``dict``, no attribute-access sugar)
        # so callers can rely on it for JSON and DataFrame construction.
        return [dict(row) for row in cursor.data()]
    except (AttributeError, TypeError):
        # Defensive: a statement without a result record (e.g. index DDL).
        return [dict(row) for row in cursor.fetchall()]


def query(target, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Read rows from either a raw connection or a :class:`GraphSession`.

    Callers that only hold one or the other should not have to branch. The
    distinction matters (a session is thread-safe, a connection is not), but it
    is an implementation detail for anything that merely reads.
    """
    if hasattr(target, "run") and not hasattr(target, "execute"):
        return target.run(cypher, params)
    return run(target, cypher, params)


def ensure_schema(conn: relate.Connection) -> None:
    """Create the indexes the read path depends on. Idempotent.

    ``stock_ticker_idx`` keeps ``MATCH (s:Stock {ticker: ...})`` a lookup instead
    of a scan - the agent does this on almost every query.
    """
    conn.execute(
        f"CREATE INDEX stock_ticker_idx IF NOT EXISTS FOR (s:{STOCK}) ON (s.ticker)"
    )


def index_names(conn: relate.Connection) -> list[str]:
    """Read back the indexes, for startup logging and smoke tests."""
    return [row["name"] for row in run(conn, "SHOW INDEXES")]


def counts(conn: relate.Connection) -> dict[str, int]:
    """Node counts per label, for a quick "is the graph populated?" check."""
    rows = run(conn, "MATCH (n) RETURN labels(n) AS labels, count(*) AS count")
    return {
        (row["labels"][0] if row["labels"] else "?"): row["count"] for row in rows
    }


def wipe(conn: relate.Connection) -> None:
    """Delete every node and relationship, keeping the store and its indexes.

    Used by ingestion scripts with ``--reset``. This is a node delete rather
    than a file delete on purpose: Relate holds the single-writer lease, so
    deleting the file underneath an open connection is not safe.
    """
    conn.execute("MATCH (n) DETACH DELETE n")


class GraphSession:
    """Thread-safe front door to the store, backed by one worker thread.

    Relate connections are **thread-affine** (``relate.threadsafety == 1``):
    a connection may only be used by the thread that opened it, and using it
    elsewhere raises ``ProgrammingError``. Streamlit runs every rerun on a fresh
    script thread, so a connection cached with ``@st.cache_resource`` works for
    the first question and fails on every question after it.

    The documented remedy is to own the connection on one thread and submit
    work to it. That is what this class does: the connection is created inside
    a single-worker executor and every statement is dispatched to that same
    worker, so callers can live on any thread.

    Use the module-level ``connect``/``run`` helpers instead for single-threaded
    ingestion scripts (no executor overhead, same API).
    """

    def __init__(self, *, read_only: bool = False, ensure_schema_on_open: bool = True) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="relate")
        self._closed = False
        self._conn = self._pool.submit(
            self._open, read_only, ensure_schema_on_open
        ).result()
        # Registered here, not only in the agent: any script or test that opens
        # a session leaves a native handle behind on crash, and an unclosed
        # handle aborts the interpreter during teardown with a pyo3 panic.
        atexit.register(self.close)

    @staticmethod
    def _open(read_only: bool, ensure: bool):
        conn = connect(read_only=read_only)
        if ensure:
            ensure_schema(conn)
        return conn

    def run(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute on the owning thread and return plain rows."""
        return self._pool.submit(run, self._conn, cypher, params).result()

    def counts(self) -> dict[str, int]:
        return self._pool.submit(counts, self._conn).result()

    def indexes(self) -> list[str]:
        return self._pool.submit(index_names, self._conn).result()

    def close(self) -> None:
        """Close on the owning thread. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        try:
            self._pool.submit(self._conn.close).result(timeout=10)
        except Exception:  # noqa: BLE001 - shutdown must never raise
            pass
        finally:
            # Drop the last reference to the native handle *while the
            # interpreter is still alive*, then join the worker. Without this
            # the wrapper can be collected during interpreter teardown, which
            # makes Relate's pyo3 layer abort with "The Python interpreter is
            # not initialized".
            self._conn = None  # type: ignore[assignment]
            self._pool.shutdown(wait=True)
