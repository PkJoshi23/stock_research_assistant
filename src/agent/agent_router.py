"""Step 1 + Step 2: detect intent, then choose only the relevant schema keys.

The two steps live in one module because they are one LLM call: asking the model
twice for facts about the same question doubles latency and cost, and the second
call can disagree with the first.

Intents:
  graph_query        answer from the connected graph (the main path)
  live_price_fetch   current quote, bypasses the graph
  historical_trend   EXPLICITLY DEFERRED - the graph stores no price history
  general_knowledge  not about the tracked graph at all

The ``historical_trend`` branch is a product decision, not a limitation we are
hiding: the assistant is about connected context, so it says so plainly and
offers what it can actually do.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import llm
from src.schema import SCHEMA_DICTIONARY

INTENT_GRAPH_QUERY = "graph_query"
INTENT_LIVE_PRICE = "live_price_fetch"
INTENT_HISTORICAL = "historical_trend"
INTENT_GENERAL = "general_knowledge"

VALID_INTENTS = {
    INTENT_GRAPH_QUERY,
    INTENT_LIVE_PRICE,
    INTENT_HISTORICAL,
    INTENT_GENERAL,
}

#: Tickers are matched loosely (BRK.B is valid) but never guessed: the previous
#: implementation used ``\b[A-Z]{1,5}\b``, which turned "AI stocks" into the
#: ticker "AI" and "Microsoft" into no ticker at all.
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,5}$")

_SYSTEM_PROMPT = """You are the routing layer of a financial GRAPH database agent.

Decide two things about the user's question:
1. Its INTENT - exactly one of: graph_query, live_price_fetch, historical_trend, general_knowledge.
2. Which SCHEMA KEYS are required to answer it.

Intent rules:
- "live_price_fetch": ONLY when the user wants a current/real-time price or quote
  right now (e.g. "what is NVDA trading at?", "AAPL price today").
- "historical_trend": any request for price history, charts, past performance,
  trends over time, moving averages, comparisons across dates
  (e.g. "how has NVDA done since 2020?", "show me a 5-year chart").
  This database stores NO price history, so these must be flagged, never
  answered from the graph.
- "graph_query": relationships, sectors, competitors, suppliers, executives,
  news, or any "how is X connected to Y" question.
- "general_knowledge": unrelated to the tracked companies or the graph
  (e.g. "what is a P/E ratio?", "who won the world cup?").

Machine extraction rules:
- "tickers": every stock ticker the question explicitly names or clearly refers
  to, uppercased (e.g. ["NVDA"]). Empty list if none is identifiable. Never
  invent a ticker for a company you are unsure about; use company names in
  "companies" instead.
- "companies": company names mentioned, as written.

Select ONLY the schema keys strictly necessary. Unrelated keys make the query
generator hallucinate.

Schema keys and their meaning:
{schema}

Return ONLY a JSON object, no markdown:
{{
  "intent": "graph_query",
  "relevant_schema_keys": ["stock_snapshot", "competitors"],
  "tickers": ["NVDA"],
  "companies": ["Nvidia"],
  "reasoning": "one short sentence"
}}"""


def _fallback(question: str, reason: str) -> dict:
    """Deterministic routing used when the LLM is unavailable or malformed.

    A demo should degrade to *something sensible*, not crash. The heuristics are
    deliberately conservative: anything price-history-shaped is deferred rather
    than accidentally answered as a graph query.
    """
    lowered = question.lower()
    if re.search(
        r"\b(trend|trending|history|historical|chart|since\s+(19|20)\d\d|"
        r"past\s+\d+\s+(day|week|month|year)|over\s+the\s+(past|last)|"
        r"last\s+(day|week|month|quarter|year)|yesterday|"
        r"previous\s+(close|month|quarter|year)|"
        r"moving\s+average|dividend\s+history|how\s+has\s+.+\s+performed)\b",
        lowered,
    ):
        intent = INTENT_HISTORICAL
        keys: list[str] = []
    elif re.search(r"\b(price|quote|trading|worth)\b", lowered) and re.search(
        r"\b(now|today|current|currently|right now|live)\b", lowered
    ):
        intent = INTENT_LIVE_PRICE
        keys = ["stock_snapshot"]
    else:
        intent = INTENT_GRAPH_QUERY
        keys = ["stock_snapshot"]

    return {
        "intent": intent,
        "relevant_schema_keys": keys,
        "tickers": [],
        "companies": [],
        "reasoning": reason,
    }


def _clean_tickers(raw) -> list[str]:
    """Validate and de-duplicate the model's ticker list."""
    if not isinstance(raw, list):
        return []
    seen: list[str] = []
    for value in raw:
        if not isinstance(value, str):
            continue
        ticker = value.strip().upper()
        if _TICKER_RE.match(ticker) and ticker not in seen:
            seen.append(ticker)
    return seen[:5]


def detect_intent_and_extract_schema(user_question: str) -> dict:
    """Run Steps 1 and 2 in a single call.

    Always returns a dict with the keys ``intent``, ``relevant_schema_keys``,
    ``tickers``, ``companies`` and ``reasoning``; never raises.
    """
    schema = "\n".join(
        f"- {key}: {description.strip().splitlines()[0]}"
        for key, description in SCHEMA_DICTIONARY.items()
    )

    try:
        parsed = llm.chat_json(
            system=_SYSTEM_PROMPT.format(schema=schema), prompt=user_question
        )
    except (ValueError, RuntimeError) as exc:
        return _fallback(user_question, f"LLM routing unavailable ({type(exc).__name__}).")

    intent = parsed.get("intent")
    if intent not in VALID_INTENTS:
        return _fallback(user_question, f"Model returned unknown intent {intent!r}.")

    keys = [
        key
        for key in parsed.get("relevant_schema_keys", [])
        if isinstance(key, str) and key in SCHEMA_DICTIONARY
    ]
    # A graph query with no keys cannot be generated safely; fall back to the
    # snapshot key, which every stock query needs anyway.
    if intent == INTENT_GRAPH_QUERY and not keys:
        keys = ["stock_snapshot"]

    return {
        "intent": intent,
        "relevant_schema_keys": keys,
        "tickers": _clean_tickers(parsed.get("tickers")),
        "companies": [
            name for name in parsed.get("companies", []) if isinstance(name, str)
        ][:5],
        "reasoning": str(parsed.get("reasoning", ""))[:300],
    }
