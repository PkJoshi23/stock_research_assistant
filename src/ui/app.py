"""Streamlit frontend for the pure-graph stock research assistant.

Run from the project root:
    streamlit run src/ui/app.py

The UI is deliberately thin: it owns chat state and telemetry, and delegates
every decision to ``StockResearchAgent``. That keeps the four-step pipeline
testable from the CLI (``python -m src.agent.master_agent "..."``) without a
browser in the loop.

The telemetry panel is part of the demo, not debug output: judges can watch the
intent decision, the selected mini-schema, the generated Cypher and the raw rows
for every question.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from src import __version__ as src_version, config, llm, memory
from src.agent.master_agent import HISTORICAL_MESSAGE, StockResearchAgent

st.set_page_config(page_title="Relate · Stock Research Agent", page_icon="📈", layout="wide")


@st.cache_resource
def get_agent() -> StockResearchAgent:
    """One agent (and therefore one Relate connection) per server process.

    Relate enforces a single-writer lease per store, so creating a connection
    per rerun would eventually collide with the cached one.
    """
    return StockResearchAgent()


def graph_stats(agent: StockResearchAgent) -> tuple[dict[str, int], list[str]]:
    try:
        return agent.db.counts(), agent.db.indexes()
    except Exception:  # noqa: BLE001 - stats must never break the chat
        return {}, []


agent = get_agent()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 Relate Stock Agent")
    st.caption("Connected financial context, stored in an embedded graph.")

    # Never prefill this widget with the server key. Streamlit serialises widget
    # values into the page, so a prefilled password field hands the credential to
    # every visitor - masked on screen, readable in devtools. The field is an
    # opt-in override; empty means "use the key this deployment was configured
    # with".
    if config.DEEPSEEK_API_KEY:
        st.caption("🟢 Server key configured - leave the field below empty to use it.")
    api_key = st.text_input(
        "Use your own DeepSeek API key",
        type="password",
        value="",
        help="Optional. Leave empty to use the key configured for this deployment.",
    )
    # Apply on every rerun: this is what makes a key typed here actually reach
    # steps 1-3 instead of only the final summary.
    llm.set_api_key(api_key.strip() or None)

    st.divider()
    counts, indexes = graph_stats(agent)
    st.subheader("Graph")
    if counts:
        st.metric("Nodes", sum(counts.values()))
        st.write(" · ".join(f"{label}: {count}" for label, count in sorted(counts.items())))
        st.caption(f"Indexes: {', '.join(indexes)}")
    else:
        st.warning("Graph is empty - run the ingestion scripts first.")

    st.divider()
    st.subheader("Memory")
    recent_questions = memory.recent(agent.db, limit=5)
    if recent_questions:
        for row in recent_questions:
            repeat = f" ×{row['count']}" if (row["count"] or 0) > 1 else ""
            st.caption(f"· {row['text']}{repeat}")
        interests = memory.interest_profile(agent.db, limit=3)
        if interests:
            st.caption(
                "Most asked about: "
                + ", ".join(f"{row['ticker']} ({row['questions']})" for row in interests)
            )
    else:
        st.caption("No questions yet - ask something and it will be remembered.")

    st.divider()
    st.subheader("Try")
    for sample in (
        "Which companies compete with Microsoft and in which sector?",
        "What is NVDA trading at right now?",
        "Who supplies Nvidia?",
        "How has Tesla's stock performed since 2020?",
        "What does the news say about AI chip demand?",
    ):
        st.caption(f"· {sample}")

    # Which build is actually serving this page. Streamlit Cloud rebuilds are
    # asynchronous, so "I pushed but the app is unchanged" is otherwise
    # indistinguishable from a code bug.
    st.caption(f"build v{src_version}")

# -----------------------------------------------------------------------------
# Chat state
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ask me about **relationships**: competitors, suppliers, sectors, "
                "executives, or news. I can also pull a live price.\n\n"
                "I don't do historical charts - I'm built on connected context."
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------------------------------------------------------
# Chat turn
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Ask about a company, its competitors, suppliers or news..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reasoning over the graph..."):
            result = agent.ask(prompt)

        # Persist the turn: the SEC-fetch action below is rendered outside this
        # block, and clicking it reruns the script with no chat input.
        st.session_state["last_result"] = result
        st.session_state["last_prompt"] = prompt

        if result["intent"] == "historical_trend":
            st.info(result["answer"])
        elif result["error"]:
            st.error(result["answer"])
        else:
            st.markdown(result["answer"])

        # Be explicit when a model call failed and a deterministic fallback
        # answered instead - never let degraded output pass as normal.
        if result.get("degraded"):
            st.warning(
                "Language model unavailable - this answer came from fixed graph "
                "queries. Questions outside those shapes need the model."
            )

        # Memory: a small, honest nudge when the question repeats.
        asked_before = (result.get("memory") or {}).get("asked_before", 0)
        if asked_before:
            st.caption(f"🕘 You've asked this {asked_before + 1}× — remembered in the graph.")

        # Connected-context map: a one-hop view of what the answer was built on.
        # Shown whenever entities were identified, including the live-price and
        # deflection branches - seeing the neighbourhood is the point of the demo.
        if not result.get("error") and result.get("tickers"):
            html_graph = agent.generate_graph_visualization(result["tickers"])
            if html_graph:
                st.markdown("### 🕸️ Connected Context Map")
                st.caption(
                    "Drag to rearrange · scroll to zoom · hover for details.  "
                    "🔵 stock · 🟢 sector · 🟠 executive · 🟣 news"
                )
                # st.iframe auto-detects HTML content; it replaces the
                # deprecated st.components.v1.html in Streamlit 1.58.
                st.iframe(html_graph, height=560)

        # Evidence: what the answer was actually built from.
        if result.get("sources"):
            with st.expander("📚 Sources & freshness", expanded=False):
                for source in result["sources"]:
                    if source.get("url"):
                        detail = f" — {source['detail']}" if source.get("detail") else ""
                        st.markdown(f"- [{source['label']}]({source['url']}){detail}")
                    elif source.get("detail"):
                        st.markdown(f"- {source['label']} — {source['detail']}")
                    else:
                        st.markdown(f"- {source['label']}")

        # Telemetry: the four steps, visible.
        with st.expander("🔍 Pipeline telemetry", expanded=False):
            st.markdown(f"**Intent** `{result['intent']}`")
            if result["schema_keys"]:
                st.markdown(f"**Schema keys** `{', '.join(result['schema_keys'])}`")
            if result["tickers"]:
                st.markdown(f"**Tickers** `{', '.join(result['tickers'])}`")
            if result["cypher"]:
                st.code(result["cypher"], language="cypher")
            if result["rows"]:
                st.markdown(f"**Rows** {len(result['rows'])}")
                st.dataframe(
                    pd.DataFrame(json.loads(json.dumps(result["rows"], default=str))),
                    width="stretch",
                )
            if result["error"]:
                st.markdown("**Error**")
                st.code(result["error"])
            usage = result.get("usage") or {}
            if usage:
                st.caption(
                    f"⚙️ {usage.get('calls', 0)} model call(s) · "
                    f"{usage.get('total_tokens', 0)} tokens · "
                    f"{usage.get('wall_ms', 0)} ms wall"
                )

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})


# -----------------------------------------------------------------------------
# Unknown-company action
# -----------------------------------------------------------------------------
# Rendered outside the chat-turn block on purpose: a button click reruns the
# script with an empty chat input, so a button nested in that block disappears
# before its own handler can run. The turn is persisted in session state instead.
last_result = st.session_state.get("last_result") or {}
last_prompt = st.session_state.get("last_prompt") or ""
coverage_info = last_result.get("coverage") or {}
untracked = coverage_info.get("untracked") or []

if coverage_info.get("can_fetch") and untracked:
    fetch_ticker = untracked[0]["ticker"]
    st.divider()
    st.caption(
        f"{fetch_ticker} is not in the graph yet. Reading its latest annual "
        "report costs one download and about 6 seconds."
    )
    if st.button(
        f"Fetch {fetch_ticker} from SEC and add it",
        key=f"fetch-sec-{fetch_ticker}",
    ):
        from src.ingestion import fast_sec

        with st.spinner(f"Reading {fetch_ticker}'s latest annual report..."):
            report = fast_sec.ingest(agent.db, fetch_ticker, write=True)

        if report.form:
            parts = [f"{report.form} filed {report.filed}"]
            if report.suppliers:
                parts.append(f"{len(report.suppliers)} suppliers")
            if report.subsidiaries:
                parts.append(f"{len(report.subsidiaries)} subsidiaries")
            parts.append(f"{report.elapsed_ms / 1000:.1f}s")
            st.success(f"Added {fetch_ticker}: " + " · ".join(parts))

            if report.suppliers:
                st.write("Suppliers: " + ", ".join(report.suppliers))
            elif report.evidence:
                st.caption(
                    "The filing names no suppliers, but discloses: "
                    f"\u201c{report.evidence}\u201d"
                )

            if last_prompt:
                with st.spinner("Answering again with the new data..."):
                    followup = agent.ask(last_prompt)
                st.session_state["last_result"] = followup
                st.session_state.messages.append(
                    {"role": "assistant", "content": followup["answer"]}
                )
                if followup.get("error"):
                    st.error(followup["answer"])
                else:
                    st.markdown(followup["answer"])
        else:
            reason = report.notes[0] if report.notes else "unknown reason"
            st.warning(f"I couldn't read a filing for {fetch_ticker}: {reason}")
