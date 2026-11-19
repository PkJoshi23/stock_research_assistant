"""Deterministic answers when the language model is unavailable.

The pipeline's *quality* comes from the model. Its *availability* must not.
A demo can hit a 401, a 429, an expired key or a flat network, and when it does
the graph is still local and still fully queryable. Without this module that
situation produces "I couldn't turn that into a graph query" - which looks like
the product is broken rather than the API being down.

So: recognise the handful of question shapes this assistant exists to answer,
and serve them with fixed, reviewed Cypher. No model, no network, no tokens.

Scope, deliberately narrow:
  * only shapes the demo actually asks (suppliers, subsidiaries, board,
    competitors, sector);
  * only in one company per question;
  * every answer is labelled as degraded so nobody mistakes it for the LLM path.

Anything else returns ``None`` and the caller falls through to its normal
error handling - a wrong offline guess is worse than an honest failure.
"""

from __future__ import annotations

import re
from typing import Callable

from src import relate_db, safety
from src.schema import (
    PERSON,
    REL_COMPETES_WITH,
    REL_IN_SECTOR,
    REL_OWNS,
    REL_SUPPLIES_TO,
    REL_WORKS_FOR,
    SECTOR,
    STOCK,
    SUBSIDIARY,
)

#: Notice attached to every offline answer, so the UI can be honest about it.
DEGRADED_NOTICE = (
    "_(The language model is unavailable, so this was answered with fixed "
    "graph queries. Questions outside this set need the model.)_"
)

#: Ordered: the first matching shape wins. Supplier questions are checked before
#: sector because "which suppliers are in this sector" should not become a
#: sector answer.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("subsidiaries", re.compile(r"\b(subsidiar(?:y|ies)|owns?|owned|acquisition)\b", re.I)),
    ("board", re.compile(r"\b(board(?:\s+of\s+directors)?|directors?)\b", re.I)),
    ("suppliers", re.compile(r"\b(suppl(?:y|ies|ier|iers)|vendors?|depend(?:s|ing)?\s+on)\b", re.I)),
    ("competitors", re.compile(r"\b(compet(?:e|es|ing|itor|itors|ition))\b", re.I)),
    ("sector", re.compile(r"\b(sector|industry)\b", re.I)),
]


def classify(question: str) -> str | None:
    """Return the offline shape name for a question, or ``None``."""
    for name, pattern in _PATTERNS:
        if pattern.search(question):
            return name
    return None


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# -----------------------------------------------------------------------------
# Handlers: one fixed query + one template each
# -----------------------------------------------------------------------------


def _suppliers(session, ticker: str) -> dict | None:
    rows = session.run(
        f"""
        MATCH (sup)-[:{REL_SUPPLIES_TO}]->(s:{STOCK} {{ticker: $t}})
        RETURN coalesce(sup.ticker, sup.name) AS supplier
        ORDER BY supplier
        """,
        {"t": ticker},
    )
    names = [row["supplier"] for row in rows if row["supplier"]]
    if not names:
        return None
    return {
        "rows": rows,
        "answer": f"{ticker} is supplied by {_join(names)}.",
    }


def _subsidiaries(session, ticker: str) -> dict | None:
    rows = session.run(
        f"""
        MATCH (s:{STOCK} {{ticker: $t}})-[:{REL_OWNS}]->(sub:{SUBSIDIARY})
        RETURN sub.name AS subsidiary
        ORDER BY subsidiary
        """,
        {"t": ticker},
    )
    names = [row["subsidiary"] for row in rows if row["subsidiary"]]
    if not names:
        return None
    return {
        "rows": rows,
        "answer": f"{ticker} owns {_join(names)}.",
    }


def _board(session, ticker: str) -> dict | None:
    rows = session.run(
        f"""
        MATCH (p:{PERSON})-[:{REL_WORKS_FOR}]->(s:{STOCK} {{ticker: $t}})
        WHERE toLower(p.role) = 'director'
        RETURN p.name AS director
        ORDER BY director
        """,
        {"t": ticker},
    )
    names = [row["director"] for row in rows if row["director"]]
    if not names:
        return None
    return {
        "rows": rows,
        "answer": f"The board of {ticker} includes {_join(names)}.",
    }


def _competitors(session, ticker: str) -> dict | None:
    rows = session.run(
        f"""
        MATCH (s:{STOCK} {{ticker: $t}})-[:{REL_COMPETES_WITH}]-(other:{STOCK})
        RETURN other.ticker AS ticker, other.name AS name
        ORDER BY ticker
        """,
        {"t": ticker},
    )
    names = [row["ticker"] for row in rows if row["ticker"]]
    if not names:
        return None
    return {
        "rows": rows,
        "answer": f"{ticker} competes with {_join(names)}.",
    }


def _sector(session, ticker: str) -> dict | None:
    rows = session.run(
        f"""
        MATCH (s:{STOCK} {{ticker: $t}})-[:{REL_IN_SECTOR}]->(sec:{SECTOR})
        RETURN sec.name AS sector
        """,
        {"t": ticker},
    )
    if not rows or not rows[0]["sector"]:
        return None
    sector = rows[0]["sector"]
    return {"rows": rows, "answer": f"{ticker} is in the {sector} sector."}


_HANDLERS: dict[str, Callable[..., dict | None]] = {
    "suppliers": _suppliers,
    "subsidiaries": _subsidiaries,
    "board": _board,
    "competitors": _competitors,
    "sector": _sector,
}


def answer_without_llm(
    session,
    question: str,
    resolve_ticker: Callable[[str], str | None],
) -> dict | None:
    """Answer a known question shape without any model call.

    ``resolve_ticker`` is injected rather than imported: the agent already has a
    three-step resolver (model tickers, graph names/aliases, uppercase token)
    and duplicating it here would let the two drift apart.

    Returns ``None`` when the shape is unknown, no ticker is identifiable, or the
    graph has nothing for it - all three mean "let the caller report the real
    failure" rather than inventing an answer.
    """
    shape = classify(question)
    if shape is None:
        return None

    ticker = resolve_ticker(question)
    if not ticker:
        return None

    handler = _HANDLERS.get(shape)
    if handler is None:
        return None

    try:
        result = handler(session, ticker)
    except Exception:  # noqa: BLE001 - a fallback must never be the crash source
        return None

    if not result:
        return None

    return {
        **result,
        "shape": shape,
        "ticker": ticker,
        "answer": f"{result['answer']}\n\n{DEGRADED_NOTICE}",
    }
