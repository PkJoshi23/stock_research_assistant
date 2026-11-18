"""Conversation memory, stored in the graph.

The pitch for an embedded graph database is that the agent's memory can be a
graph, not a transcript. This module is that claim in ~120 lines: each question
becomes a ``(:Query)`` node, repeated questions accumulate a counter, and the
companies a question was about get ``ABOUT`` edges.

What that buys:

  * "you've asked about NVDA three times" - personalisation without a vector
    store or a session ledger;
  * a sidebar of recent questions that survives a Streamlit restart;
  * a queryable trail: *which companies has this user actually cared about?*

Deliberately not stored: the answer text. Memory should record what was asked,
not duplicate the graph's own content - and storing model output would make the
memory the least trustworthy data in the database.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re

from src.schema import QUERY, REL_ABOUT, REL_IN_SECTOR, SECTOR, STOCK

_WS = re.compile(r"\s+")


def query_id(question: str) -> str:
    """Stable identity for a question.

    Normalising (lower-case, collapse whitespace, strip trailing punctuation)
    means "Who supplies Nvidia?" and "who supplies nvidia " are the same memory,
    which is what a user expects from "have I asked this before?".

    A hash keeps ids short and avoids putting arbitrary text in a lookup key; the
    full text is stored as a property for display.
    """
    normalised = _WS.sub(" ", question.strip().lower()).strip(" ?!.")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def remember(session, question: str, intent: str, tickers: list[str]) -> dict:
    """Record a question and the companies it touched. Never raises.

    Memory is a side effect; if it fails, the answer must still be returned. The
    caller treats the return value as best-effort metadata.
    """
    info = {"asked_before": 0, "last_asked": None, "id": query_id(question)}
    try:
        qid = info["id"]
        now = dt.datetime.now()

        existing = session.run(
            f"MATCH (q:{QUERY} {{id: $id}}) RETURN q.count AS count", {"id": qid}
        )
        asked_before = int(existing[0]["count"] or 0) if existing else 0

        session.run(
            f"""
            MERGE (q:{QUERY} {{id: $id}})
            SET q.text    = $text,
                q.intent  = $intent,
                q.count   = coalesce(q.count, 0) + 1,
                q.last_at = $now
            """,
            {"id": qid, "text": question.strip()[:300], "intent": intent, "now": now},
        )
        # First sighting records when; MERGE + SET above would overwrite it.
        if asked_before == 0:
            session.run(
                f"MATCH (q:{QUERY} {{id: $id}}) SET q.first_at = $now", {"id": qid, "now": now}
            )

        for ticker in tickers[:6]:
            session.run(
                f"""
                MATCH (q:{QUERY} {{id: $id}}), (s:{STOCK} {{ticker: $ticker}})
                MERGE (q)-[:{REL_ABOUT}]->(s)
                """,
                {"id": qid, "ticker": ticker},
            )

        info["asked_before"] = asked_before
        info["last_asked"] = str(now)
    except Exception:  # noqa: BLE001 - memory must never break an answer
        return info
    return info


def recent(session, limit: int = 8) -> list[dict]:
    """Most recent distinct questions, newest first."""
    try:
        return session.run(
            f"""
            MATCH (q:{QUERY})
            RETURN q.text AS text, q.intent AS intent, q.count AS count,
                   q.last_at AS last_at
            ORDER BY q.last_at DESC
            LIMIT $limit
            """,
            {"limit": int(limit)},
        )
    except Exception:  # noqa: BLE001
        return []


def ticker_history(session, ticker: str, limit: int = 5) -> list[dict]:
    """Recent questions that touched one company."""
    try:
        return session.run(
            f"""
            MATCH (q:{QUERY})-[:{REL_ABOUT}]->(s:{STOCK} {{ticker: $t}})
            RETURN q.text AS text, q.last_at AS last_at
            ORDER BY q.last_at DESC
            LIMIT $limit
            """,
            {"t": ticker, "limit": int(limit)},
        )
    except Exception:  # noqa: BLE001
        return []


def interest_profile(session, limit: int = 5) -> list[dict]:
    """Which companies this user's questions have been about, most first.

    ``sum(q.count)`` rather than ``count(q)``: asking the same question three
    times is three expressions of interest, not one. This is the query that
    justifies storing memory as a graph rather than a log file - the answer is a
    traversal, not a scan over parsed text.
    """
    try:
        return session.run(
            f"""
            MATCH (q:{QUERY})-[:{REL_ABOUT}]->(s:{STOCK})
            OPTIONAL MATCH (s)-[:{REL_IN_SECTOR}]->(sec:{SECTOR})
            RETURN s.ticker AS ticker, sum(q.count) AS questions,
                   coalesce(sec.name, '') AS sector
            ORDER BY questions DESC, ticker ASC
            LIMIT $limit
            """,
            {"limit": int(limit)},
        )
    except Exception:  # noqa: BLE001
        return []
