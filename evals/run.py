"""Offline / live evaluation harness for the 4-step text-to-Cypher pipeline.

Run from the repo root:

    .\\.venv\\Scripts\\python.exe -m evals.run          # offline: no key, no network
    .\\.venv\\Scripts\\python.exe -m evals.run --live   # opt-in: real DeepSeek

Offline mode is deliberately narrow and honest about it:

* it stubs ``src.llm.chat`` / ``chat_json`` / ``get_client`` and the embedding
  entry points in-process, so it cannot spend a token or open a socket;
* it points ``config.DB_PATH`` at a throwaway store under ``evals/.eval-tmp``
  and refuses to run if that resolves to ``data/stock_graph.rl``;
* it exercises the deterministic router fallback, the read-only guard, schema
  validation and the reviewed reference queries against a real (temporary)
  embedded Relate store built from ``tests/conftest.seed_demo_graph``.

It does NOT run the text-to-Cypher generator with a model; that needs ``--live``
(bounded to a small routing subset). See ``evals/README.md`` for what the two
modes prove and what they do not.

Exit codes: 0 = all gated checks passed, 1 = a real failure, 2 = the harness
could not start (missing dependency, unsafe store path, ...).
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

#: Started before the pipeline modules are imported, so the reported runtime is
#: the wall time a user actually waits for, not just the check phase.
_HARNESS_STARTED = time.perf_counter()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVALS_DIR = Path(__file__).resolve().parent
REAL_DB = ROOT / "data" / "stock_graph.rl"

from evals.questions import EVAL_FIXTURE, QUESTIONS, Question  # noqa: E402
from src import config, embeddings, llm, relate_db  # noqa: E402
from src.agent import agent_router, cypher_executor, cypher_generator  # noqa: E402
from src.schema import (  # noqa: E402
    PERSON,
    REL_COMPETES_WITH,
    REL_IN_SECTOR,
    REL_OWNS,
    REL_SUPPLIES_TO,
    REL_WORKS_FOR,
    SECTOR,
    STOCK,
    SUBSIDIARY,
    SUPPLIER,
    SCHEMA_DICTIONARY,
    mini_schema,
)

INTENT_GRAPH_QUERY = agent_router.INTENT_GRAPH_QUERY
INTENT_HISTORICAL = agent_router.INTENT_HISTORICAL
INTENT_LIVE_PRICE = agent_router.INTENT_LIVE_PRICE
INTENT_GENERAL = agent_router.INTENT_GENERAL
VALID_INTENTS = agent_router.VALID_INTENTS

#: Categories the dataset is required to cover.
REQUIRED_CATEGORIES = {
    "competitors",
    "suppliers",
    "subsidiaries",
    "board",
    "sector",
    "live_price",
    "historical",
    "general",
}

#: Declared labels / relationship types, used to check that a reference query
#: only uses schema-sanctioned entities.
DECLARED_LABELS = {STOCK, SECTOR, PERSON, SUBSIDIARY, SUPPLIER}
DECLARED_RELS = {
    REL_IN_SECTOR,
    REL_COMPETES_WITH,
    REL_SUPPLIES_TO,
    REL_WORKS_FOR,
    REL_OWNS,
}

#: Statements that must be rejected before reaching the engine.
WRITE_CORPUS = (
    "CREATE (:Stock {ticker: 'X'})",
    "MERGE (s:Stock {ticker: 'X'})",
    "MATCH (s:Stock) SET s.name = 'x'",
    "MATCH (s:Stock) DETACH DELETE s",
    "MATCH (s:Stock) REMOVE s.name",
    "DROP INDEX stock_ticker_idx",
    "LOAD CSV FROM 'file:///x' AS row RETURN row",
    "CALL dbms.procedures()",
    "CALL apoc.periodic.iterate('x', 'y', {})",
    "CALL db.schema.visualization()",
    "merge (s:Stock {ticker: 'X'})",
)

#: Read-only statements that contain write keywords as data, not as statements.
READ_CORPUS = (
    "MATCH (s:Stock) WHERE s.name = 'Create Inc' RETURN s.ticker AS ticker",
    "MATCH (p:Person)-[:WORKS_FOR]->(s:Stock) "
    "WHERE toLower(p.role) = 'director' RETURN p.name AS director",
)

_LABEL_RE = re.compile(r"\(\s*(?:[A-Za-z_]\w*\s*)?:\s*([A-Za-z_]\w*)")
_REL_RE = re.compile(r"\[\s*(?:[A-Za-z_]\w*\s*)?:\s*([A-Za-z_]\w*)")


class HarnessError(RuntimeError):
    """The harness itself cannot run; not a pipeline failure."""


@dataclass(frozen=True)
class QuestionResult:
    qid: str
    category: str
    status: str  # PASS | FAIL | GAP
    question: str
    detail: str


@dataclass(frozen=True)
class AuxResult:
    name: str
    ok: bool
    detail: str


# -----------------------------------------------------------------------------
# Process isolation: no model, no network, no embeddings
# -----------------------------------------------------------------------------


@contextmanager
def _offline_llm() -> Iterator[dict[str, int]]:
    """Replace every model/embedding entry point for the duration of the run.

    The stubs raise, so a code path that accidentally needs a model fails
    loudly and is reported instead of silently reaching the network. The
    counters let the harness prove the router actually used the stubbed path
    and that no client or embedding model was requested.
    """
    counters = {"chat_json": 0, "chat": 0, "get_client": 0, "embed": 0}

    def chat_json(*args: Any, **kwargs: Any) -> dict:
        counters["chat_json"] += 1
        raise RuntimeError("offline eval: language model disabled")

    def chat(*args: Any, **kwargs: Any) -> str:
        counters["chat"] += 1
        raise RuntimeError("offline eval: language model disabled")

    def get_client(*args: Any, **kwargs: Any) -> Any:
        counters["get_client"] += 1
        raise AssertionError("offline eval: network client requested")

    def embed(*args: Any, **kwargs: Any) -> Any:
        counters["embed"] += 1
        raise AssertionError("offline eval: embedding model requested")

    targets: tuple[tuple[Any, str, Callable[..., Any]], ...] = (
        (llm, "chat", chat),
        (llm, "chat_json", chat_json),
        (llm, "get_client", get_client),
        (embeddings, "embed_text", embed),
        (embeddings, "embed_texts", embed),
    )
    originals = [(module, name, getattr(module, name)) for module, name, _ in targets]
    for module, name, replacement in targets:
        setattr(module, name, replacement)
    try:
        yield counters
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


# -----------------------------------------------------------------------------
# Throwaway store
# -----------------------------------------------------------------------------


def _load_seed_demo_graph() -> Callable[[Callable[..., Any]], None]:
    """Import ``seed_demo_graph`` from tests/conftest.py without importing the
    repo's ``tests`` package.

    ``tests`` has no ``__init__.py`` and this machine has an unrelated
    site-packages package also called ``tests``, so a normal import can resolve
    to the wrong module. Loading the file by path is deterministic and keeps the
    helper read-only (nothing under tests/ is modified).
    """
    conftest = ROOT / "tests" / "conftest.py"
    if not conftest.is_file():
        raise HarnessError(f"seeding helper not found: {conftest}")
    spec = importlib.util.spec_from_file_location("_evals_seed_conftest", conftest)
    if spec is None or spec.loader is None:
        raise HarnessError(f"could not load {conftest}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise HarnessError(
            "loading tests/conftest.py requires pytest (a dev dependency); "
            f"install it into the venv: {exc}"
        ) from exc
    seed = getattr(module, "seed_demo_graph", None)
    if not callable(seed):
        raise HarnessError("tests/conftest.py no longer defines seed_demo_graph")
    return seed


def _apply_fixture(session: relate_db.GraphSession) -> None:
    for cypher, params in EVAL_FIXTURE:
        session.run(cypher, params)


@contextmanager
def _temp_store() -> Iterator[tuple[relate_db.GraphSession, Path]]:
    """Open a seeded store inside ``evals/.eval-tmp`` and clean it up after.

    The production store is never opened: the path is redirected before the
    session is created and the redirect is restored afterwards.
    """
    real = REAL_DB.resolve()
    original_path = config.DB_PATH
    tmp_root = EVALS_DIR / ".eval-tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=str(tmp_root)))
    session: relate_db.GraphSession | None = None
    try:
        # Redirect BEFORE any connection exists: the production store is never
        # opened, not even briefly, and the redirect is restored afterwards.
        config.DB_PATH = run_dir / "eval_graph.rl"
        if config.DB_PATH.resolve() == real:
            raise HarnessError("internal error: eval store path resolved to production")
        try:
            session = relate_db.GraphSession()
            _load_seed_demo_graph()(session.run)
            _apply_fixture(session)
        except HarnessError:
            raise
        except Exception as exc:  # noqa: BLE001 - a setup problem is not a pipeline result
            raise HarnessError(
                f"could not build the eval store: {type(exc).__name__}: {exc}"
            ) from exc
        yield session, run_dir
    finally:
        config.DB_PATH = original_path
        if session is not None:
            session.close()
        shutil.rmtree(run_dir, ignore_errors=True)
        try:
            tmp_root.rmdir()  # only succeeds when the last run removed its dir
        except OSError:
            pass


def _counts(session: relate_db.GraphSession) -> tuple[int, int]:
    nodes = session.run("MATCH (n) RETURN count(n) AS c")[0]["c"]
    rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    return int(nodes), int(rels)


# -----------------------------------------------------------------------------
# Static / dataset checks
# -----------------------------------------------------------------------------


def _check_dataset() -> AuxResult:
    problems: list[str] = []
    ids = [q.id for q in QUESTIONS]
    if len(ids) != len(set(ids)):
        problems.append("question ids are not unique")
    if not 20 <= len(QUESTIONS) <= 35:
        problems.append(f"expected ~25 questions, found {len(QUESTIONS)}")

    categories = {q.category for q in QUESTIONS}
    missing = REQUIRED_CATEGORIES - categories
    if missing:
        problems.append(f"categories not covered: {sorted(missing)}")
    intents = {q.expected_intent for q in QUESTIONS}
    if intents != VALID_INTENTS:
        problems.append(
            f"dataset must cover all intents; missing {sorted(VALID_INTENTS - intents)}"
        )

    for q in QUESTIONS:
        if q.expected_intent not in VALID_INTENTS:
            problems.append(f"{q.id}: unknown expected intent {q.expected_intent!r}")
        bad_keys = set(q.expected_schema_keys) - set(SCHEMA_DICTIONARY)
        if bad_keys:
            problems.append(f"{q.id}: expected keys not in schema: {sorted(bad_keys)}")
        if q.expected_intent == INTENT_GRAPH_QUERY and not q.reference_cypher:
            problems.append(f"{q.id}: graph question has no reference query")
        if q.reference_cypher is None and (
            q.expected_row_count is not None
            or q.expected_columns
            or q.expected_cells
            or q.forbidden_cells
            or q.min_rows
        ):
            problems.append(f"{q.id}: row expectations without a reference query")
        if q.reference_cypher is not None and not q.execute_reference and (
            q.expected_row_count is not None or q.expected_columns or q.expected_cells
        ):
            problems.append(
                f"{q.id}: would check rows of a reference query that is not executed"
            )
        if q.offline_gap and q.expected_intent != INTENT_GENERAL:
            problems.append(f"{q.id}: offline_gap is only meaningful for general questions")

    live = [q for q in QUESTIONS if q.live]
    if not 3 <= len(live) <= 12:
        problems.append(f"--live subset should be small (3-12), found {len(live)}")

    detail = f"{len(QUESTIONS)} questions across {len(categories)} categories; {len(live)} live"
    if not problems:
        return AuxResult("dataset integrity", True, detail)
    return AuxResult("dataset integrity", False, "; ".join(problems))


def _check_grounding() -> AuxResult:
    """Every reference query may only use declared schema entities, and those
    entities must be described by the question's expected schema keys.

    This catches dataset drift when ``src/schema.py`` changes: a reference query
    that references a renamed label, or a question whose expected key no longer
    describes the relationship it needs.
    """
    problems: list[str] = []
    for q in QUESTIONS:
        if not q.reference_cypher:
            continue
        labels = {match.group(1) for match in _LABEL_RE.finditer(q.reference_cypher)}
        rels = {match.group(1) for match in _REL_RE.finditer(q.reference_cypher)}
        unknown = (labels - DECLARED_LABELS) | (rels - DECLARED_RELS)
        if unknown:
            problems.append(f"{q.id}: undeclared entities {sorted(unknown)}")
        if q.expected_schema_keys:
            description = mini_schema(sorted(q.expected_schema_keys))
            undescribed = {name for name in labels | rels if name not in description}
            if undescribed:
                problems.append(
                    f"{q.id}: entities not described by expected keys "
                    f"{sorted(q.expected_schema_keys)}: {sorted(undescribed)}"
                )
    if not problems:
        return AuxResult("schema grounding", True, "references use only declared, key-described entities")
    return AuxResult("schema grounding", False, "; ".join(problems))


# -----------------------------------------------------------------------------
# Guard checks
# -----------------------------------------------------------------------------


def _check_guard_corpus() -> AuxResult:
    problems: list[str] = []
    for cypher in WRITE_CORPUS:
        if cypher_executor.guard_read_only(cypher) is None:
            problems.append(f"guard ALLOWED a write: {cypher!r}")
    read_corpus = list(READ_CORPUS) + [
        q.reference_cypher for q in QUESTIONS if q.reference_cypher
    ]
    for cypher in read_corpus:
        problem = cypher_executor.guard_read_only(cypher)
        if problem:
            problems.append(f"guard BLOCKED a read ({problem}): {cypher!r}")
    if not problems:
        return AuxResult(
            "read-only guard corpus",
            True,
            f"{len(WRITE_CORPUS)} writes blocked, {len(read_corpus)} reads allowed",
        )
    return AuxResult("read-only guard corpus", False, "; ".join(problems))


def _check_guard_wired(session: relate_db.GraphSession) -> AuxResult:
    """The guard must run inside Step 4, before the engine sees the statement."""
    leak = "CREATE (:EvalLeak {id: 1})"
    result = cypher_executor.execute_cypher_with_retry(
        session, leak, "eval guard check", ["stock_snapshot"], max_retries=0
    )
    problems: list[str] = []
    if "non-read-only" not in result.get("error", ""):
        problems.append(
            "a write reached execute_cypher_with_retry without being blocked "
            f"(error={result.get('error')!r})"
        )
    if result.get("rows"):
        problems.append("a blocked write returned rows")
    if not problems:
        return AuxResult("guard inside Step 4", True, "write refused before execution")
    return AuxResult("guard inside Step 4", False, "; ".join(problems))


# -----------------------------------------------------------------------------
# Per-question offline checks
# -----------------------------------------------------------------------------


def _row_problems(q: Question, rows: list[dict]) -> list[str]:
    problems: list[str] = []
    if q.expected_row_count is not None and len(rows) != q.expected_row_count:
        problems.append(
            f"expected {q.expected_row_count} row(s), got {len(rows)}"
        )
    if len(rows) < q.min_rows:
        problems.append(f"expected at least {q.min_rows} row(s), got {len(rows)}")
    if q.expected_columns:
        if not rows:
            problems.append("no rows returned, so expected columns are unverified")
        else:
            missing = set(q.expected_columns) - set(rows[0].keys())
            if missing:
                problems.append(f"missing column(s) {sorted(missing)}")
    for column, value in q.expected_cells:
        if not any(str(row.get(column)) == value for row in rows):
            problems.append(f"expected {column}={value!r} in the result")
    for column, value in q.forbidden_cells:
        if any(str(row.get(column)) == value for row in rows):
            problems.append(f"{column}={value!r} must not appear in the result")
    return problems


def _check_question(q: Question, session: relate_db.GraphSession) -> QuestionResult:
    problems: list[str] = []
    evidence: list[str] = []

    routing = agent_router.detect_intent_and_extract_schema(q.question)
    intent = routing.get("intent")
    keys = routing.get("relevant_schema_keys") or []
    evidence.append(f"intent={intent}")
    evidence.append(f"keys={list(keys)}")

    # Schema validation is gated for every question, gap or not.
    if not isinstance(keys, list):
        problems.append(f"schema keys are not a list: {keys!r}")
        keys = []
    invalid = [key for key in keys if key not in SCHEMA_DICTIONARY]
    if invalid:
        problems.append(f"invalid schema key(s) {invalid}")
    if intent == INTENT_HISTORICAL and keys:
        problems.append(
            "historical question returned schema keys; deferral must need no graph"
        )
    if intent == INTENT_GRAPH_QUERY and not keys:
        problems.append("graph_query returned no schema keys")

    status = "PASS"
    gap_detail = ""

    # Intent: product truth, except where the deterministic fallback provably
    # cannot reach it. Those are tracked as gaps, never counted as passes.
    if q.offline_gap:
        if intent != q.expected_intent:
            status = "GAP"
            gap_detail = (
                f"expected {q.expected_intent}, fallback returned {intent}; "
                f"{q.offline_gap}"
            )
        else:
            evidence.append("fallback now meets the product intent")
    elif intent != q.expected_intent:
        problems.append(f"intent {intent!r} != expected {q.expected_intent!r}")

    # The non-graph intents must be routed away from the graph, deterministically
    # and before any model call.
    if intent == INTENT_HISTORICAL:
        generated = cypher_generator.generate_cypher_query(
            q.question, list(keys), INTENT_HISTORICAL
        )
        if generated.get("action") != "skip" or generated.get("cypher") is not None:
            problems.append(
                "historical intent did not skip Cypher generation "
                f"(action={generated.get('action')!r})"
            )
        evidence.append("generator=skip (deferred)")
    elif intent == INTENT_LIVE_PRICE:
        generated = cypher_generator.generate_cypher_query(
            q.question, list(keys), INTENT_LIVE_PRICE
        )
        if generated.get("action") != "skip" or generated.get("cypher") is not None:
            problems.append(
                "live-price intent did not skip Cypher generation "
                f"(action={generated.get('action')!r})"
            )
        evidence.append("generator=skip (tool path)")

    # Reference query: read-only, then executed through Step 4 (guard included).
    if q.reference_cypher:
        guard_problem = cypher_executor.guard_read_only(q.reference_cypher)
        if guard_problem:
            problems.append(f"reference query rejected by guard: {guard_problem}")
        elif q.execute_reference:
            execution = cypher_executor.execute_cypher_with_retry(
                session,
                q.reference_cypher,
                q.question,
                sorted(q.expected_schema_keys) or ["stock_snapshot"],
                max_retries=0,  # one attempt, no LLM repair, no token spend
            )
            if execution.get("error"):
                problems.append(f"reference query failed: {execution['error']}")
            elif execution.get("attempts") != 1:
                problems.append(
                    f"reference query needed {execution.get('attempts')} attempts"
                )
            else:
                rows = execution.get("rows") or []
                problems.extend(_row_problems(q, rows))
                evidence.append(f"rows={len(rows)}")
        else:
            evidence.append("guard only (execution needs an embedding model)")

    if problems:
        return QuestionResult(q.id, q.category, "FAIL", q.question, "; ".join(problems))
    if status == "GAP":
        return QuestionResult(q.id, q.category, "GAP", q.question, gap_detail)
    return QuestionResult(q.id, q.category, "PASS", q.question, " ".join(evidence))


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def _safe_aux(name: str, fn: Callable[..., AuxResult], *args: Any) -> AuxResult:
    """Run one harness check; a harness bug becomes a failed check, not a crash."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - keep the table printable
        return AuxResult(name, False, f"harness error: {type(exc).__name__}: {exc}")


def _truncate(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def _print_table(results: list[QuestionResult], id_width: int) -> None:
    print(
        f"{'ID':<{id_width}}  {'CATEGORY':<13}  {'RESULT':<5}  "
        f"{'QUESTION':<46}  EVIDENCE"
    )
    print("-" * 118)
    for result in results:
        print(
            f"{result.qid:<{id_width}}  {result.category:<13}  "
            f"{result.status:<5}  {_truncate(result.question, 46):<46}  "
            f"{_truncate(result.detail, 70)}"
        )


def _print_aux(aux: list[AuxResult]) -> None:
    print("Harness checks")
    for item in aux:
        print(f"  {'PASS' if item.ok else 'FAIL':<4}  {item.name:<24}  {item.detail}")


def _print_summary(
    results: list[QuestionResult], aux: list[AuxResult], elapsed: float
) -> int:
    passed = [r for r in results if r.status == "PASS"]
    failed = [r for r in results if r.status == "FAIL"]
    gaps = [r for r in results if r.status == "GAP"]
    gated = len(results) - len(gaps)

    print()
    print("-" * 118)
    print("Offline summary")
    print(f"  questions in dataset      : {len(results)}")
    print(f"  gated offline             : {gated}")
    print(f"  passed                    : {len(passed)}")
    print(f"  failed                    : {len(failed)}")
    print(f"  known gaps (not gated)    : {len(gaps)}  (needs the LLM; run --live)")
    gated_rate = (len(passed) / gated * 100.0) if gated else 0.0
    product_rate = (len(passed) / len(results) * 100.0) if results else 0.0
    print(f"  offline gated pass rate   : {len(passed)}/{gated} = {gated_rate:.1f}%")
    print(
        f"  product-intent pass rate  : {len(passed)}/{len(results)} = "
        f"{product_rate:.1f}%  (gaps count as not passing)"
    )
    print(f"  runtime                   : {elapsed:.1f}s")

    print("By category:")
    categories = sorted({r.category for r in results})
    for category in categories:
        rows = [r for r in results if r.category == category]
        cat_pass = sum(1 for r in rows if r.status == "PASS")
        cat_fail = sum(1 for r in rows if r.status == "FAIL")
        cat_gap = sum(1 for r in rows if r.status == "GAP")
        if cat_gap:
            print(
                f"  {category:<14} {cat_pass}/{len(rows)} passed, "
                f"{cat_gap} gap(s)"
            )
        else:
            print(f"  {category:<14} {cat_pass}/{len(rows)} passed, {cat_fail} failed")

    if gaps:
        print()
        print("Known offline gaps (tracked, NOT counted as passes)")
        for gap in gaps:
            print(f"  {gap.qid}: {gap.detail}")

    aux_failed = [item for item in aux if not item.ok]
    ok = not failed and not aux_failed
    print()
    print(f"RESULT: {'PASS' if ok else 'FAIL'} (exit {0 if ok else 1})")
    return 0 if ok else 1


# -----------------------------------------------------------------------------
# Modes
# -----------------------------------------------------------------------------


def run_offline() -> int:
    print("=" * 118)
    print("AI Stock Research Assistant - OFFLINE evaluation")
    print("=" * 118)
    print(f"Repository : {ROOT}")
    print(f"Real store : {REAL_DB}  (never opened)")
    print("Mode       : offline - LLM/embedding entry points stubbed in-process; "
          "no network, no API key")

    results: list[QuestionResult] = []
    aux: list[AuxResult] = []
    with _offline_llm() as counters:
        try:
            with _temp_store() as (session, run_dir):
                print(f"Eval store : {run_dir / 'eval_graph.rl'}  (temporary)")
                print("=" * 118)
                counts_before = _counts(session)

                aux.append(_safe_aux("dataset integrity", _check_dataset))
                aux.append(_safe_aux("schema grounding", _check_grounding))
                aux.append(_safe_aux("read-only guard corpus", _check_guard_corpus))
                aux.append(_safe_aux("guard inside Step 4", _check_guard_wired, session))

                for question in QUESTIONS:
                    try:
                        results.append(_check_question(question, session))
                    except Exception as exc:  # noqa: BLE001 - one bad check must not abort
                        results.append(
                            QuestionResult(
                                question.id,
                                question.category,
                                "FAIL",
                                question.question,
                                f"harness error: {type(exc).__name__}: {exc}",
                            )
                        )

                counts_after = _counts(session)
                if counts_before == counts_after:
                    aux.append(
                        AuxResult(
                            "store unchanged",
                            True,
                            f"{counts_after[0]} nodes / {counts_after[1]} relationships "
                            "before and after",
                        )
                    )
                else:
                    aux.append(
                        AuxResult(
                            "store unchanged",
                            False,
                            f"before={counts_before}, after={counts_after}",
                        )
                    )
        except HarnessError as exc:
            print(f"HARNESS ERROR: {exc}")
            return 2

    isolation_ok = (
        counters["chat_json"] == len(QUESTIONS)
        and counters["chat"] == 0
        and counters["get_client"] == 0
        and counters["embed"] == 0
    )
    aux.append(
        AuxResult(
            "LLM / network isolation",
            isolation_ok,
            f"{counters['chat_json']} routing calls served by the stub; "
            f"{counters['get_client']} client calls, "
            f"{counters['chat']} chat calls, "
            f"{counters['embed']} embedding calls",
        )
    )

    _print_table(results, max((len(r.qid) for r in results), default=20))
    print()
    _print_aux(aux)
    elapsed = time.perf_counter() - _HARNESS_STARTED
    return _print_summary(results, aux, elapsed)


def _live_preflight() -> bool:
    """Fail clearly instead of silently falling back when no key is configured.

    ``detect_intent_and_extract_schema`` swallows a missing key (the fallback is
    a feature), which would make a keyless live run look like a model run. So
    the client is constructed explicitly first; constructing it does not call
    the API.
    """
    try:
        llm.get_client()
    except Exception as exc:  # noqa: BLE001 - report, do not guess
        print("LIVE MODE UNAVAILABLE: " + str(exc))
        print(
            "Live mode needs DEEPSEEK_API_KEY. Offline mode needs no key and no "
            "network: run without --live."
        )
        return False
    return True


def run_live() -> int:
    live_questions = [q for q in QUESTIONS if q.live]
    print("=" * 118)
    print("AI Stock Research Assistant - LIVE routing evaluation")
    print("=" * 118)
    print(f"Questions  : {len(live_questions)} (one real DeepSeek call each)")
    print("Scope      : routing intent + schema keys only. No graph access, no "
          "yfinance, no Cypher generation.")
    print("=" * 118)

    if not _live_preflight():
        return 2

    results: list[QuestionResult] = []
    for question in live_questions:
        try:
            routing = agent_router.detect_intent_and_extract_schema(question.question)
        except Exception as exc:  # noqa: BLE001 - a failed call is a failed check
            results.append(
                QuestionResult(
                    question.id,
                    question.category,
                    "FAIL",
                    question.question,
                    f"router raised: {type(exc).__name__}: {exc}",
                )
            )
            continue

        problems: list[str] = []
        intent = routing.get("intent")
        keys = set(routing.get("relevant_schema_keys") or [])
        reasoning = str(routing.get("reasoning") or "")
        fell_back = "LLM routing unavailable" in reasoning or "unknown intent" in reasoning
        if fell_back:
            # Coincidentally matching the fallback is NOT a model pass.
            problems.append(f"router fell back to heuristics ({reasoning})")
        if intent != question.expected_intent:
            problems.append(f"intent {intent!r} != expected {question.expected_intent!r}")
        missing = set(question.expected_schema_keys) - keys
        if missing:
            problems.append(f"missing schema key(s) {sorted(missing)}")
        invalid = keys - set(SCHEMA_DICTIONARY)
        if invalid:
            problems.append(f"invalid schema key(s) {sorted(invalid)}")

        detail = f"intent={intent}; keys={sorted(keys)}"
        results.append(
            QuestionResult(
                question.id,
                question.category,
                "FAIL" if problems else "PASS",
                question.question,
                "; ".join(problems) if problems else detail,
            )
        )

    _print_table(results, max((len(r.qid) for r in results), default=20))
    passed = sum(1 for r in results if r.status == "PASS")
    failed = len(results) - passed
    elapsed = time.perf_counter() - _HARNESS_STARTED
    print()
    print("-" * 118)
    print(f"LIVE summary: {passed}/{len(results)} passed, {failed} failed, {elapsed:.1f}s")
    print(
        "Reminder: this measures routing only. It does not prove that generated "
        "Cypher is correct or that answers are good."
    )
    print(f"RESULT: {'PASS' if not failed else 'FAIL'} (exit {0 if not failed else 1})")
    return 0 if not failed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline evaluation harness for the stock research agent. "
            "Default mode: no API key, no network."
        )
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="opt in to real DeepSeek routing calls for a small subset of questions",
    )
    args = parser.parse_args(argv)
    if args.live:
        return run_live()
    return run_offline()


if __name__ == "__main__":
    raise SystemExit(main())
