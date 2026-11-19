"""Step 4: the read-only guard and the bounded repair loop.

The guard is a safety control, not a prompt suggestion: a hallucinated write must
never reach a graph that is being demoed live.
"""

from __future__ import annotations

import pytest

from src import config, llm
from src.agent import cypher_executor


class FakeSession:
    """Stands in for GraphSession without a real store."""

    def __init__(self, failures: int = 0, rows: list[dict] | None = None) -> None:
        self.failures = failures
        self.calls = 0
        self.queries: list[str] = []
        self.rows = rows if rows is not None else [{"ticker": "NVDA"}]

    def run(self, cypher: str, params: dict | None = None) -> list[dict]:
        self.calls += 1
        self.queries.append(cypher)
        if self.calls <= self.failures:
            raise RuntimeError(f"engine failure {self.calls}")
        return self.rows


@pytest.mark.parametrize(
    "cypher",
    [
        "CREATE (:Stock {ticker: 'X'})",
        "MERGE (s:Stock {ticker: 'X'})",
        "MATCH (s:Stock) SET s.name = 'x'",
        "MATCH (s:Stock) DETACH DELETE s",
        "MATCH (s:Stock) REMOVE s.name",
        "DROP INDEX stock_ticker_idx",
        "LOAD CSV FROM 'file:///x' AS row RETURN row",
        "CALL dbms.procedures()",
        "merge (s:Stock {ticker: 'X'})",
    ],
)
def test_guard_blocks_writes(cypher: str):
    assert cypher_executor.guard_read_only(cypher) is not None


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (s:Stock) RETURN s.ticker AS ticker",
        "MATCH (s:Stock) WHERE s.name = 'Create Inc' RETURN s.ticker AS ticker",
        # A property value containing a keyword is data, not a statement.
        "MATCH (s:Stock) WHERE s.name CONTAINS 'delete' RETURN s.name AS name",
    ],
)
def test_guard_allows_reads(cypher: str):
    assert cypher_executor.guard_read_only(cypher) is None


def test_guard_runs_before_any_execution():
    session = FakeSession()
    result = cypher_executor.execute_cypher_with_retry(
        session, "CREATE (:X)", "question"
    )
    assert "non-read-only" in result["error"]
    assert session.calls == 0


def test_build_params_is_empty_without_placeholders():
    assert cypher_executor.build_params("MATCH (s:Stock) RETURN s.ticker", "q") == {}


def test_successful_execution_does_not_call_the_llm(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("LLM must not be called for a successful query")

    monkeypatch.setattr(llm, "chat", explode)
    session = FakeSession()
    result = cypher_executor.execute_cypher_with_retry(
        session, "MATCH (s:Stock) RETURN s.ticker AS ticker", "q"
    )
    assert result["error"] == ""
    assert result["rows"] == [{"ticker": "NVDA"}]
    assert result["attempts"] == 1


def test_repair_loop_recovers_after_failures(monkeypatch):
    monkeypatch.setattr(
        llm, "chat", lambda *a, **k: "MATCH (s:Stock) RETURN s.ticker AS ticker"
    )
    session = FakeSession(failures=2)
    result = cypher_executor.execute_cypher_with_retry(
        session, "MATCH (s RETURN broken", "q", max_retries=2
    )
    assert result["error"] == ""
    assert result["attempts"] == 3
    assert result["rows"] == [{"ticker": "NVDA"}]


def test_repair_loop_is_bounded(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "MATCH (s:Stock) RETURN s.ticker")
    session = FakeSession(failures=99)
    result = cypher_executor.execute_cypher_with_retry(
        session, "MATCH (s:Stock) RETURN s.ticker", "q", max_retries=2
    )
    assert result["error"].startswith("Query failed after 3 attempt(s)")
    assert session.calls == 3


def test_repair_prompt_contains_the_engine_error(monkeypatch):
    seen: dict = {}

    def capture(prompt, **kwargs):
        seen["prompt"] = prompt
        return "MATCH (s:Stock) RETURN s.ticker AS ticker"

    monkeypatch.setattr(llm, "chat", capture)
    session = FakeSession(failures=1)
    cypher_executor.execute_cypher_with_retry(session, "MATCH (s:Stock RETURN s", "q", max_retries=1)
    assert "engine failure 1" in seen["prompt"]


def test_llm_failure_during_repair_returns_the_original_error(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("API down")

    monkeypatch.setattr(llm, "chat", boom)
    session = FakeSession(failures=99)
    result = cypher_executor.execute_cypher_with_retry(
        session, "MATCH (s:Stock) RETURN s.ticker", "q", max_retries=2
    )
    assert "engine failure" in result["error"]
    assert "repair failed" in result["error"]


def test_default_retry_budget_comes_from_config(monkeypatch):
    monkeypatch.setattr(config, "MAX_CYPHER_RETRIES", 1)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "MATCH (s:Stock) RETURN s.ticker")
    session = FakeSession(failures=99)
    result = cypher_executor.execute_cypher_with_retry(
        session, "MATCH (s:Stock) RETURN s.ticker", "q"
    )
    assert result["attempts"] == 2  # 1 initial + 1 retry
