"""Step 4: execute in Relate, and repair the query when it fails.

Bounded by ``config.MAX_CYPHER_RETRIES``. Two safeguards the previous version
lacked:

* **A read-only guard.** The generation prompt says "never write", but a prompt
  is not a control. Any statement containing CREATE/MERGE/SET/DELETE/... is
  rejected before it reaches the engine, so a hallucinated write cannot mutate
  the graph mid-demo.
* **Shared error reporting.** Relate errors carry ``neo4j_code``, ``stage`` and
  ``retryable``; feeding those back verbatim gives the model far more to work
  with than a bare exception string.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import relate

from src import config, llm
from src.schema import mini_schema

#: Anything that can mutate the store. Checked against the query with string
#: literals removed, so a company called "Create Inc" cannot trip the guard.
_WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|FOREACH|"
    r"LOAD\s+CSV|CALL\s+dbms|CALL\s+apoc|CALL\s+db\.schema)\b",
    re.IGNORECASE,
)
_STRING_LITERALS = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")

_FIX_PROMPT = """The Cypher query below failed to execute in Relate (an embedded,
Neo4j-compatible engine). Fix it so it runs.

User question: {question}

Failed query:
{cypher}

Error: {error}

Relevant schema (use nothing outside this):
{schema}

Rules: read-only query only; explicit aliased RETURN columns; case-insensitive
string matching; LIMIT {limit} unless aggregating.
Return ONLY the corrected raw Cypher query - no markdown, no explanation."""


def guard_read_only(cypher: str) -> str | None:
    """Return a human-readable reason if the query is not read-only."""
    without_strings = _STRING_LITERALS.sub("''", cypher)
    match = _WRITE_KEYWORDS.search(without_strings)
    if match:
        return (
            f"Blocked a non-read-only statement ({match.group(0).upper()}). "
            "The agent may only read from the graph."
        )
    return None


def build_params(cypher: str, user_question: str) -> dict:
    """Supply the placeholders the generator is allowed to use.

    There are none: the research graph takes no vector parameters. The function
    stays because the executor's signature is part of the pipeline contract, and
    a future parameterised query belongs here rather than inline.
    """
    return {}


def execute_cypher_with_retry(
    session,
    cypher: str,
    user_question: str,
    schema_keys: list[str] | None = None,
    max_retries: int | None = None,
) -> dict:
    """Execute, and retry with an LLM repair on failure.

    ``session`` is a :class:`src.relate_db.GraphSession` (thread-safe) rather
    than a raw connection, so the caller may live on any thread.

    Returns ``{"rows", "cypher", "attempts", "error"}`` - never raises, because
    every caller renders the failure as an answer.
    """
    retries = config.MAX_CYPHER_RETRIES if max_retries is None else max_retries
    schema = mini_schema(schema_keys or ["stock_snapshot"])
    current = cypher.strip()
    last_error = ""

    for attempt in range(1, retries + 2):
        problem = guard_read_only(current)
        if problem:
            return {"rows": [], "cypher": current, "attempts": attempt, "error": problem}

        try:
            rows = session.run(current, build_params(current, user_question))
            return {"rows": rows, "cypher": current, "attempts": attempt, "error": ""}
        except relate.RelateError as exc:
            # Prefer the engine's own classification over the message text.
            code = getattr(exc, "neo4j_code", None) or type(exc).__name__
            stage = getattr(exc, "stage", None)
            last_error = f"{code}{f' ({stage})' if stage else ''}: {exc}"
        except Exception as exc:  # noqa: BLE001 - driver-level failures
            last_error = f"{type(exc).__name__}: {exc}"

        print(f"  ! attempt {attempt} failed: {last_error}")

        if attempt > retries:
            break

        try:
            repaired = llm.chat(
                _FIX_PROMPT.format(
                    question=user_question,
                    cypher=current,
                    error=last_error,
                    schema=schema,
                    limit=config.DEFAULT_RESULT_LIMIT,
                ),
                temperature=0.0,
            )
            current = llm.extract_cypher(repaired)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{last_error} | repair failed: {type(exc).__name__}: {exc}"
            break

    return {
        "rows": [],
        "cypher": current,
        "attempts": retries + 1,
        "error": f"Query failed after {retries + 1} attempt(s): {last_error}",
    }
