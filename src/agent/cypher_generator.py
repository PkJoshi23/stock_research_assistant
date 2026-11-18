"""Step 3: translate the question into Cypher using ONLY the mini-schema.

The grounding contract: whatever Step 2 selected is the entire universe the
model may reference. If a label or relationship is not in the prompt, the model
cannot invent it - which is the whole reason the schema is a dictionary of small,
selectable blocks rather than one big prompt.

The only placeholder the model is allowed to emit is ``$query_vector`` (plus
``$k``) for vector search; the executor fills those in, because embedding the
question requires Python, not Cypher.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config, llm
from src.agent.agent_router import (
    INTENT_GRAPH_QUERY,
    INTENT_HISTORICAL,
    INTENT_LIVE_PRICE,
)
from src.schema import mini_schema

#: Few-shot examples, keyed by the schema key that makes them legal.
#:
#: Rules written as prose are followed inconsistently; a complete query is
#: copied. Each example is only shown when its key survived Step 2, so an
#: example can never teach a label the mini-schema does not contain.
#:
#: The supplier example is the important one. It exists because the model kept
#: writing `(sup:Stock)-[:SUPPLIES_TO]->...`, which silently drops the unlisted
#: vendors (a `Supplier` node, not a `Stock`), and because `sup.ticker` is null
#: for those vendors. The unlabeled left node plus `coalesce` is the whole fix.
_FEW_SHOT: dict[str, list[tuple[str, str]]] = {
    "corporate_structure": [
        (
            "Which companies does Nvidia depend on as suppliers?",
            "MATCH (sup)-[:SUPPLIES_TO]->(s:Stock {ticker: 'NVDA'})\n"
            "RETURN coalesce(sup.ticker, sup.name) AS supplier\n"
            "ORDER BY supplier\n"
            "LIMIT 50",
        ),
        (
            "What subsidiaries does Nvidia own?",
            "MATCH (s:Stock {ticker: 'NVDA'})-[:OWNS]->(sub:Subsidiary)\n"
            "RETURN sub.name AS subsidiary\n"
            "LIMIT 50",
        ),
    ],
    "executives": [
        (
            "Who is on Nvidia's board of directors?",
            "MATCH (p:Person)-[:WORKS_FOR]->(s:Stock {ticker: 'NVDA'})\n"
            "WHERE toLower(p.role) = 'director'\n"
            "RETURN p.name AS director\n"
            "LIMIT 50",
        ),
    ],
    "competitors": [
        (
            "Which companies compete with Microsoft?",
            "MATCH (s:Stock)-[:COMPETES_WITH]-(c:Stock)\n"
            "WHERE toLower(s.ticker) = toLower('MSFT')\n"
            "RETURN c.ticker AS ticker, c.name AS name\n"
            "LIMIT 50",
        ),
    ],
}

#: Supplier guidance, injected only when a supplier-bearing key was selected.
#:
#: Kept out of the static rules on purpose: the mini-schema contract says a
#: prompt describes only what Step 2 selected, and a "no supplier questions" run
#: should not be told about suppliers at all.
_SUPPLIER_RULE = """Supplier queries: the supplier end of SUPPLIES_TO is POLYMORPHIC
(a Stock or a Supplier). Leave that end UNLABELED and return
`coalesce(sup.ticker, sup.name)` - unlisted vendors have no ticker property."""

#: Bound the prompt: every example costs tokens on every request.
_MAX_EXAMPLES = 3

_SYSTEM_PROMPT = """You translate a natural-language question into ONE read-only
Cypher query for an embedded Neo4j-compatible graph database.

Allowed schema - do not use any label, relationship or property that is not
listed here:
{schema}

Hard rules:
1. READ-ONLY. Only MATCH / OPTIONAL MATCH / WITH / WHERE / RETURN / ORDER BY /
   LIMIT / CALL db.index.vector.queryNodes. Never CREATE, MERGE, SET, DELETE,
   REMOVE, DROP or LOAD CSV.
2. Return explicit, aliased columns (e.g. `RETURN s.ticker AS ticker,
   s.name AS name`), never whole nodes - the result is rendered as a table.
3. String comparisons are case-insensitive, and a company can be named by its
   ticker, its full name, or an alias - always match all three:
   `WHERE toLower(s.ticker) = toLower('tsmc')
      OR toLower(s.name) CONTAINS toLower('tsmc')
      OR any(a IN s.aliases WHERE toLower(a) CONTAINS toLower('tsmc'))`
4. Always end with `LIMIT {limit}` unless the query is a pure aggregation
   (count/sum/avg).
5. Competitor edges are stored in one direction only - match them undirected.
6. This database has NO price history. If asked about past prices or trends,
   return nothing useful rather than inventing a pattern.
7. For semantic news search, copy the vector-search template from the schema
   verbatim and use the $query_vector and $k placeholders exactly.

{supplier_rule}

Worked examples for this schema subset. When a question matches one, copy its
shape exactly and substitute only the entity names:

{examples}

Output: the raw Cypher query only. No markdown fences, no explanation, no
trailing prose."""


def _supplier_rule(schema_keys: list[str]) -> str:
    """Include supplier guidance only when a supplier-bearing key was selected."""
    supplier_keys = {"supply_chain", "corporate_structure"}
    return _SUPPLIER_RULE if supplier_keys & set(schema_keys) else ""


def _render_examples(schema_keys: list[str]) -> str:
    """Collect the examples belonging to the selected schema keys.

    Order follows ``schema_keys`` so the most relevant key's example comes
    first, and the total is capped to keep the prompt bounded.
    """
    rendered: list[str] = []
    for key in schema_keys:
        for question, cypher in _FEW_SHOT.get(key, []):
            rendered.append(f"Q: {question}\nCypher:\n{cypher}")
            if len(rendered) >= _MAX_EXAMPLES:
                return "\n\n".join(rendered)
    return "\n\n".join(rendered) if rendered else "(no example for this subset)"


def generate_cypher_query(
    user_question: str, schema_keys: list[str], intent: str
) -> dict:
    """Generate Cypher for a graph query.

    Returns a dict with ``action`` set to:
      * ``"graph_query"``  - ``cypher`` holds the query
      * ``"skip"``         - the intent is handled without the graph
      * ``"error"``        - generation failed; ``reasoning`` says why
    """
    if intent == INTENT_LIVE_PRICE:
        return {
            "action": "skip",
            "cypher": None,
            "reasoning": "Live price intent: answered by the yfinance tool, not the graph.",
        }
    if intent == INTENT_HISTORICAL:
        return {
            "action": "skip",
            "cypher": None,
            "reasoning": "Historical trend intent: the graph stores no price history.",
        }
    if intent != INTENT_GRAPH_QUERY:
        return {
            "action": "skip",
            "cypher": None,
            "reasoning": f"Intent {intent!r} does not require a graph query.",
        }

    system = _SYSTEM_PROMPT.format(
        schema=mini_schema(schema_keys),
        limit=config.DEFAULT_RESULT_LIMIT,
        examples=_render_examples(schema_keys),
        supplier_rule=_supplier_rule(schema_keys),
    )

    try:
        raw = llm.chat(user_question, system=system, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 - network/auth failures are reported
        return {"action": "error", "cypher": None, "reasoning": f"{type(exc).__name__}: {exc}"}

    cypher = llm.extract_cypher(raw)
    if not cypher:
        return {"action": "error", "cypher": None, "reasoning": "Model returned an empty query."}

    return {
        "action": "graph_query",
        "cypher": cypher,
        "reasoning": f"Generated from {len(schema_keys)} schema key(s).",
    }
