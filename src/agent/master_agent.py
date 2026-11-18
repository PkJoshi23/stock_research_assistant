"""The orchestrator: one question in, one grounded answer out.

Pipeline:
    ask()
      ├─ Step 1+2  detect_intent_and_extract_schema()
      ├─ dispatch by intent
      │    ├─ historical_trend → designed refusal (no LLM, no graph)
      │    ├─ live_price_fetch → yfinance snapshot + graph context
      │    ├─ general_knowledge → short LLM answer, labelled as such
      │    └─ graph_query → Step 3 generate → Step 4 execute → summarise
      └─ structured result for the UI (intent, schema, cypher, rows, answer)

Every branch returns the same dict shape so the Streamlit layer never needs to
know which path produced the answer.
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config, entities, llm, memory, provenance, quotes, relate_db, safety, visualization
from src.agent import offline
from src.agent.agent_router import (
    INTENT_GENERAL,
    INTENT_GRAPH_QUERY,
    INTENT_HISTORICAL,
    INTENT_LIVE_PRICE,
    detect_intent_and_extract_schema,
)
from src.agent.cypher_executor import execute_cypher_with_retry
from src.agent.cypher_generator import generate_cypher_query
from src.schema import STOCK

#: The product decision, stated to the user rather than hidden. Keep it warm:
#: a refusal that explains the trade-off reads as confidence, not limitation.
HISTORICAL_MESSAGE = (
    "I don't do historical price charts - this assistant is built around "
    "*connected context* rather than time series, so the graph deliberately "
    "stores no price history.\n\n"
    "What I can do instead:\n"
    "- pull the **current** live price for any ticker\n"
    "- show who a company **competes with**, **supplies**, or is **supplied by**\n"
    "- find **news** about a company, or search news by meaning across the graph"
)

_SUMMARY_SYSTEM = """You are a financial research analyst answering from a graph
database. Use ONLY the rows provided. Quote the actual numbers and names in them.

Rules:
- Be concise: 1-4 sentences, or a short bullet list for several entities.
- Never invent a company, number or relationship that is not in the rows.
- If the rows are empty, say that the graph has no matching data for the
  question, and suggest a related question it could answer.
- Do not mention "rows", "JSON" or the database schema; speak about the
  companies."""


def _jsonable(rows: list[dict]) -> list[dict]:
    """Rows contain datetimes; make them JSON-safe before prompting/logging."""
    return json.loads(json.dumps(rows, default=str))


class StockResearchAgent:
    """Stateful facade over the pipeline. Owns one Relate connection."""

    def __init__(self, api_key: str | None = None) -> None:
        if api_key:
            llm.set_api_key(api_key)
        # GraphSession owns the connection on a single worker thread, which is
        # what makes caching this agent across Streamlit reruns safe: Relate
        # connections are thread-affine and each rerun is a new thread.
        self.db = relate_db.GraphSession()
        # Relate is a native extension. If the process exits while a connection
        # is still open, its finaliser can run after the interpreter is torn
        # down and abort with a pyo3 "interpreter is not initialized" panic.
        # Closing from atexit keeps teardown quiet (Streamlit, notebooks, tests).
        self._closed = False
        atexit.register(self.close)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the session once. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self.db.close()

    def __enter__(self) -> "StockResearchAgent":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- public API --------------------------------------------------------

    def ask(self, user_question: str) -> dict:
        """Answer one question. Never raises; failures come back in ``error``.

        Wraps the pipeline with telemetry (token usage, wall time) and memory
        (which questions have been asked before). Both are recorded even when
        the answer fails, because a failed question is still a question the user
        asked.
        """
        started = time.perf_counter()
        llm.start_trace()
        try:
            result = self._ask(user_question)
        finally:
            usage = llm.finish_trace()
            usage["wall_ms"] = round((time.perf_counter() - started) * 1000, 1)

        result["usage"] = usage
        result["memory"] = self._remember(user_question, result)
        return result

    def _remember(self, question: str, result: dict) -> dict:
        """Best-effort memory write: never allowed to break an answer."""
        tickers = result.get("tickers") or self._tickers_from(question)
        try:
            return memory.remember(
                self.db, question, result.get("intent", ""), list(tickers or [])
            )
        except Exception:  # noqa: BLE001
            return {"asked_before": 0, "last_asked": None}

    def _ask(self, user_question: str) -> dict:
        """The pipeline itself, without telemetry or memory."""
        try:
            routing = detect_intent_and_extract_schema(user_question)
        except Exception as exc:  # noqa: BLE001 - routing must not crash the app
            return self._result(
                intent=INTENT_GRAPH_QUERY,
                answer="I could not classify that question. Please rephrase it.",
                error=f"{type(exc).__name__}: {exc}",
            )

        result = self._result(
            intent=routing["intent"],
            schema_keys=routing["relevant_schema_keys"],
            tickers=routing["tickers"],
        )

        try:
            if routing["intent"] == INTENT_HISTORICAL:
                result["answer"] = HISTORICAL_MESSAGE
            elif routing["intent"] == INTENT_LIVE_PRICE:
                result.update(
                    self._answer_live_price(user_question, routing["tickers"])
                )
            elif routing["intent"] == INTENT_GENERAL:
                result.update(self._answer_general(user_question))
            else:
                result.update(
                    self._answer_from_graph(
                        user_question, routing["relevant_schema_keys"]
                    )
                )
        except Exception as exc:  # noqa: BLE001 - surface, do not crash the UI
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["answer"] = (
                "Something went wrong while answering that. "
                f"({type(exc).__name__}) - check the logs for details."
            )
        return result

    # -- intent branches ---------------------------------------------------

    def _answer_live_price(self, question: str, tickers: list[str]) -> dict:
        """Current quote via yfinance, enriched with graph context.

        The price is live and never stored; the context (sector, rivals) is what
        makes this a graph product rather than a quote widget.
        """
        ticker = self._resolve_ticker(question, tickers)
        if not ticker:
            return {
                "answer": (
                    "I couldn't tell which ticker you mean. Name it explicitly "
                    "(for example \"NVDA\") and I'll pull the live price."
                ),
                "error": "",
            }

        quote = quotes.fetch(ticker)
        if quote is None:
            return {
                "answer": f"I couldn't retrieve a live quote for **{ticker}** right now.",
                "error": "no price returned by yfinance",
            }

        context = self._graph_context(ticker)
        answer = f"**{ticker}** is trading at **${quote.price:,.2f} {quote.currency}** right now."
        if context["name"]:
            answer += f" ({context['name']}"
            answer += f", {context['sector']})" if context["sector"] else ")"
        if context["competitors"]:
            answer += f"\n\nIn my graph it competes with: {', '.join(context['competitors'])}."
        if context["suppliers"]:
            answer += f"\nIt is supplied by: {', '.join(context['suppliers'])}."
        answer += "\n\nLive quote, not stored - the graph keeps current snapshots only."

        return {
            "answer": answer,
            "rows": [],
            "tickers": [ticker],
            "sources": provenance.live_quote_source(
                ticker, quote.price, quote.currency, quote.at
            )
            + provenance.collect(self.db, ticker, []),
        }

    def _answer_general(self, question: str) -> dict:
        """Off-graph questions: answer briefly, and label the source honestly."""
        answer = llm.chat(
            question,
            system=(
                "You are a concise financial-analysis assistant. Answer in at "
                "most three sentences, and state clearly that this answer does "
                "not come from the user's stock graph."
            ),
            temperature=0.3,
        )
        return {"answer": answer, "rows": []}

    def _answer_from_graph(self, question: str, schema_keys: list[str]) -> dict:
        """The main path: generate Cypher, execute with repair, summarise.

        Both halves have a deterministic fallback: if generation or execution
        fails (typically the model being unreachable), the question is retried
        against fixed Cypher in ``offline.py`` and the answer is flagged
        ``degraded``. A local graph should not disappear because an API did.
        """
        generated = generate_cypher_query(question, schema_keys, INTENT_GRAPH_QUERY)
        if generated["action"] != "graph_query":
            fallback = self._offline_answer(question)
            if fallback:
                return fallback
            return {
                "answer": (
                    "I couldn't turn that into a graph query. "
                    f"({generated['reasoning']})"
                ),
                "error": generated["reasoning"],
            }

        execution = execute_cypher_with_retry(
            self.db, generated["cypher"], question, schema_keys
        )
        rows = execution["rows"]

        if execution["error"]:
            fallback = self._offline_answer(question)
            if fallback:
                return fallback
            return {
                "cypher": execution["cypher"],
                "rows": rows,
                "answer": (
                    "I couldn't run that query against the graph even after "
                    f"retrying.\n\n`{execution['error']}`"
                ),
                "error": execution["error"],
            }

        if not rows:
            # Deterministic and free: no point paying an LLM to say "nothing".
            return {
                "cypher": execution["cypher"],
                "rows": [],
                "answer": (
                    "The graph has no data matching that. I track a small set of "
                    "companies and their competitors, suppliers, executives and "
                    "news - try asking about one of those relationships."
                ),
            }

        # Retrieved rows are untrusted (headlines come from the web). Sanitise
        # them and mark them as data before they enter the prompt.
        safe_rows = safety.sanitize_rows(_jsonable(rows[:40]))
        try:
            summary = llm.chat(
                f"Question: {question}\n\n"
                + safety.wrap_untrusted(json.dumps(safe_rows, indent=2), "GRAPH ROWS"),
                system=f"{_SUMMARY_SYSTEM}\n\n{safety.UNTRUSTED_DATA_RULE}",
                temperature=0.2,
            )
        except Exception:  # noqa: BLE001 - model down after a *successful* query
            # The rows are real and already retrieved; render them rather than
            # throwing away a good result because the summariser failed.
            return {
                "cypher": execution["cypher"],
                "rows": rows,
                "answer": self._render_rows(rows),
                "sources": self._sources(question, rows),
                "degraded": True,
            }
        return {
            "cypher": execution["cypher"],
            "rows": rows,
            "answer": summary,
            "sources": self._sources(question, rows),
        }

    def _sources(self, question: str, rows: list[dict]) -> list[dict]:
        """Evidence for an answer: rows, filings, snapshot time, headlines."""
        try:
            ticker = self._resolve_ticker(question, [])
            return provenance.collect(self.db, ticker, rows)
        except Exception:  # noqa: BLE001 - provenance must never break an answer
            return []

    def _offline_answer(self, question: str) -> dict | None:
        """Last-resort deterministic answer for a known question shape."""
        result = offline.answer_without_llm(
            self.db, question, lambda q: self._resolve_ticker(q, [])
        )
        if not result:
            return None
        return {
            "answer": result["answer"],
            "rows": result["rows"],
            "cypher": None,
            "sources": self._sources(question, result["rows"]),
            "degraded": True,
        }

    @staticmethod
    def _render_rows(rows: list[dict], limit: int = 12) -> str:
        """Render rows as text when no model is available to summarise them."""
        lines = []
        for row in rows[:limit]:
            cells = ", ".join(
                f"{key}: {safety.sanitize(str(value), max_chars=200)}"
                for key, value in row.items()
            )
            lines.append(f"- {cells}")
        more = "" if len(rows) <= limit else f"\n\n(+{len(rows) - limit} more rows)"
        return (
            "Here is what the graph returned (the language model is unavailable, "
            "so this is unsummarised):\n\n" + "\n".join(lines) + more
            + f"\n\n{offline.DEGRADED_NOTICE}"
        )

    # -- helpers -----------------------------------------------------------

    def _companies(self) -> list[dict]:
        """One graph read of the tracked companies, shared by both matchers."""
        return entities.load_companies(self.db)

    def _resolve_ticker(self, question: str, tickers: list[str]) -> str | None:
        """Pick the ticker the question is about.

        Matching is delegated to :mod:`src.entities` so the agent, the news
        loader and the SEC loader share one set of rules. This method used to
        carry a private third implementation plus its own graph read.
        """
        return entities.best_match(question, self._companies(), hint=tickers)

    def _graph_context(self, ticker: str) -> dict:
        """Small graph neighbourhood used to enrich the live-price answer."""
        rows = self.db.run(
            f"""
            MATCH (s:{STOCK} {{ticker: $t}})
            OPTIONAL MATCH (s)-[:IN_SECTOR]->(sec:Sector)
            RETURN s.name AS name, coalesce(sec.name, '') AS sector
            """,
            {"t": ticker},
        )
        name = rows[0]["name"] if rows else ""
        sector = rows[0]["sector"] if rows else ""

        competitors = [
            row["ticker"]
            for row in self.db.run(
                f"""
                MATCH (s:{STOCK} {{ticker: $t}})-[:COMPETES_WITH]-(other:{STOCK})
                RETURN other.ticker AS ticker
                """,
                {"t": ticker},
            )
        ]
        suppliers = [
            row["supplier"]
            for row in self.db.run(
                f"""
                MATCH (sup)-[:SUPPLIES_TO]->(s:{STOCK} {{ticker: $t}})
                RETURN coalesce(sup.ticker, sup.name) AS supplier
                """,
                {"t": ticker},
            )
        ]
        return {
            "name": name,
            "sector": sector,
            "competitors": competitors,
            "suppliers": suppliers,
        }

    def generate_graph_visualization(self, tickers_or_question: str | list[str]) -> str:
        """Return a self-contained pyvis HTML document for the relevant subgraph.

        Accepts either the tickers already extracted in Step 1 (the UI passes
        ``result["tickers"]``) or a raw question, in which case tickers are
        resolved against the graph - no extra LLM call for a decision the router
        has already made.

        Returns ``""`` when there is nothing to draw, so callers can simply test
        truthiness. Returns HTML rather than a file path: writing a file per
        question leaks temp files, needs cleanup, and collides once two sessions
        ask at the same time.
        """
        tickers = self._tickers_from(tickers_or_question)
        
        if not tickers:
            return ""
        subgraph = visualization.fetch_subgraph(
            self.db, tickers, news_per_stock=2
        )
        if not subgraph["nodes"]:
            return ""
        return visualization.build_pyvis_html(subgraph)

    def _tickers_from(self, source: str | list[str]) -> list[str]:
        """Coerce a ticker list or a free-text question into tickers.

        Free text goes through the shared scan policy, so a question, a headline
        and a filing name are all matched by the same rules.
        """
        if isinstance(source, (list, tuple, set)):
            return [str(t).strip().upper() for t in source if str(t).strip()][:6]

        matches = entities.resolve(str(source), self._companies(), policy="scan")
        return [match.ticker for match in matches][:6]

    @staticmethod
    def _result(**overrides) -> dict:
        """Canonical result shape; every branch starts from this."""
        result = {
            "intent": "",
            "schema_keys": [],
            "tickers": [],
            "cypher": None,
            "rows": [],
            "answer": "",
            "error": "",
            #: Evidence for the answer (rows, filings, snapshot time, headlines).
            "sources": [],
            #: Token/latency telemetry for this question (see src/llm.py).
            "usage": {},
            #: Whether this question has been asked before, and by which id.
            "memory": {},
            #: True when the answer came from deterministic fallbacks because a
            #: model call failed. The UI shows a banner for these.
            "degraded": False,
        }
        result.update(overrides)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the stock research agent.")
    parser.add_argument("question", nargs="?", help="question to ask")
    args = parser.parse_args()

    question = args.question or "Which companies compete with Microsoft and what sector are they in?"

    with StockResearchAgent() as agent:
        result = agent.ask(question)

    print(f"\nQuestion : {question}")
    print(f"Intent   : {result['intent']}  keys={result['schema_keys']}")
    if result["cypher"]:
        print(f"Cypher   :\n{result['cypher']}")
    if result["error"]:
        print(f"Error    : {result['error']}")
    print(f"\nAnswer   :\n{result['answer']}")
    return 0 if not result["error"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
