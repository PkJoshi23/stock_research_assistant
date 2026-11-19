"""Where an answer came from.

An answer that cannot be audited is a guess with better grammar. This module
turns what the agent already holds into a short evidence list: how many rows came
back, whether SEC filings contributed
relationships, and when the snapshot was last refreshed.

It fetches nothing. Everything here is read from the graph the answer was built
from, so the provenance can never disagree with the answer.
"""

from __future__ import annotations

from src import relate_db
from src.schema import PERSON, REL_SUPPLIES_TO, REL_WORKS_FOR, STOCK

#: Cap on headline citations - a long list stops being readable.

def _officer_edges(session, ticker: str) -> int:
    rows = session.run(
        f"""
        MATCH (p:{PERSON})-[r:{REL_WORKS_FOR}]->(s:{STOCK} {{ticker: $t}})
        WHERE r.source = 'DEF 14A'
        RETURN count(r) AS n
        """,
        {"t": ticker},
    )
    return rows[0]["n"] if rows else 0


def _filing_supplier_edges(session, ticker: str) -> int:
    rows = session.run(
        f"""
        MATCH (sup)-[r:{REL_SUPPLIES_TO}]->(s:{STOCK} {{ticker: $t}})
        WHERE r.source = '10-K' OR r.confirmedBy = '10-K'
        RETURN count(r) AS n
        """,
        {"t": ticker},
    )
    return rows[0]["n"] if rows else 0


def _snapshot_timestamp(session, ticker: str) -> str | None:
    rows = session.run(
        f"MATCH (s:{STOCK} {{ticker: $t}}) RETURN s.updated_at AS updated",
        {"t": ticker},
    )
    if not rows or rows[0]["updated"] is None:
        return None
    return str(rows[0]["updated"])


def collect(
    session,
    ticker: str | None,
    rows: list[dict],
    *,
    intent: str = "graph_query",
) -> list[dict]:
    """Build the evidence list for one answer.

    Ordering is deliberate: the rows themselves first (what the answer is made
    of), then filings (the most authoritative relationships)
    (the most perishable).
    """
    sources: list[dict] = []

    if rows:
        sources.append(
            {
                "type": "graph",
                "label": f"{len(rows)} row{'s' if len(rows) != 1 else ''} from the Relate graph",
            }
        )

    if ticker:
        officers = _officer_edges(session, ticker)
        if officers:
            sources.append(
                {
                    "type": "filing",
                    "label": f"{officers} officer/director link{'s' if officers != 1 else ''} from the DEF 14A proxy statement",
                }
            )

        suppliers = _filing_supplier_edges(session, ticker)
        if suppliers:
            sources.append(
                {
                    "type": "filing",
                    "label": f"{suppliers} supplier relationship{'s' if suppliers != 1 else ''} confirmed by the 10-K",
                }
            )

        snapshot = _snapshot_timestamp(session, ticker)
        if snapshot:
            sources.append({"type": "snapshot", "label": f"market snapshot as of {snapshot}"})

    return sources


def live_quote_source(ticker: str, price: float, currency: str, at: str) -> list[dict]:
    """Evidence line for a live quote: it was fetched, at a moment, not stored."""
    return [
        {
            "type": "live_quote",
            "label": f"{ticker} {price:,.2f} {currency}".strip(),
            "detail": f"fetched live at {at}; not persisted to the graph",
        }
    ]
