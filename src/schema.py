"""Single source of truth for the graph schema.

Ingestion writes against these names; the agent is told about them. Keeping one
module prevents the classic failure mode where an LLM is told about ``Company``
while the loader wrote ``Stock`` - the query is valid Cypher, returns zero rows,
and nothing errors.

Two rules for editing this file:
  1. Every label/relationship/property added here must also be written by an
     ingestion script, or the agent will confidently query data that cannot
     exist.
  2. Descriptions are prompt text. They are read by the LLM, not just humans, so
     they must be exact about direction, cardinality and property types.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------

STOCK = "Stock"
SECTOR = "Sector"
PERSON = "Person"
SUBSIDIARY = "Subsidiary"
SUPPLIER = "Supplier"
QUERY = "Query"  # the user's question history, written by src/memory.py

# -----------------------------------------------------------------------------
# Relationship types
# -----------------------------------------------------------------------------

REL_IN_SECTOR = "IN_SECTOR"        # (:Stock)-[:IN_SECTOR]->(:Sector)
REL_COMPETES_WITH = "COMPETES_WITH"  # (:Stock)-[:COMPETES_WITH]->(:Stock)
REL_SUPPLIES_TO = "SUPPLIES_TO"    # (:Stock)-[:SUPPLIES_TO]->(:Stock)  supplier -> customer
REL_WORKS_FOR = "WORKS_FOR"        # (:Person)-[:WORKS_FOR]->(:Stock)

# From SEC filings (see src/ingestion/ingest_sec_relationships.py)
REL_OWNS = "OWNS"  # (:Stock)-[:OWNS]->(:Subsidiary)

# Conversation memory (see src/memory.py)
REL_ABOUT = "ABOUT"  # (:Query)-[:ABOUT]->(:Stock)

# NOTE: suppliers use ONE relationship type regardless of whether the supplier
# is a tracked stock or an untracked vendor: (:Stock|Supplier)-[:SUPPLIES_TO]->(:Stock).
# An earlier split (Stock)-[:SUPPLIES_TO] vs (Supplier)-[:CRITICAL_SUPPLIER_FOR]
# forced every supplier query into a two-branch union, and the model reliably
# dropped one branch - answering "who supplies X?" with half the answer.

# -----------------------------------------------------------------------------
# The schema dictionary handed to the routing/extraction step (Step 2)
# -----------------------------------------------------------------------------
#
# Step 2 selects a subset of these keys; Step 3 receives only that subset. This
# is what keeps the generation prompt grounded: the model never sees the whole
# graph, so it cannot invent a label it was never shown.
#
# The graph is research context only: companies, their relationships, and the
# current snapshot. There is no news and no price history.

SCHEMA_DICTIONARY: dict[str, str] = {
    "stock_snapshot": f"""
        Node: {STOCK}
        Properties:
          ticker        (string, unique)  - e.g. "NVDA"
          name          (string)          - e.g. "NVIDIA Corporation"
          aliases       (list of strings) - names users actually type, e.g. ["Nvidia"]
          sector        (string)          - e.g. "Technology"
          industry      (string)          - e.g. "Semiconductors"
          market_cap    (float, USD)
          pe_ratio      (float)
          current_price (float, USD)
          updated_at    (datetime)        - when the snapshot was last refreshed
        Resolve a company by matching ANY of these three, case-insensitively:
          toLower(s.ticker) = toLower('tsmc')
          OR toLower(s.name) CONTAINS toLower('tsmc')
          OR any(a IN s.aliases WHERE toLower(a) CONTAINS toLower('tsmc'))
        IMPORTANT: only CURRENT values are stored. This graph has no price
        history, so questions about past prices cannot be answered from it.
    """,
    "sector_membership": f"""
        Relationship: (:{STOCK})-[:{REL_IN_SECTOR}]->(:{SECTOR} {{name: string}})
        Use to find the sector of a stock, or every stock inside a sector.
    """,
    "competitors": f"""
        Relationship: (:{STOCK})-[:{REL_COMPETES_WITH}]->(:{STOCK})
        The edge is stored in one direction only. Treat it as symmetric: use
        an undirected pattern, e.g. (:Stock)-[:COMPETES_WITH]-(:Stock).
    """,
    "supply_chain": f"""
        Relationship: (:{STOCK})-[:{REL_SUPPLIES_TO}]->(:{STOCK})
        Direction is supplier -> customer. To find who supplies a company, match
        (:Stock)-[:{REL_SUPPLIES_TO}]->(c:Stock); to find a supplier's customers,
        match (s:Stock)-[:{REL_SUPPLIES_TO}]->(:Stock).
    """,
    "executives": f"""
        Node: {PERSON}
        Properties:
          name   (string, unique)
          role   (string) - values in use: "CEO", "CFO", "Director", "Officer"
        Relationship: (:{PERSON})-[:{REL_WORKS_FOR}]->(:{STOCK})
        Use to find who runs a company, or which companies a person is tied to.
        People here come from SEC proxy statements (DEF 14A): named executive
        officers keep their title, and board members carry role "Director".
        Note: a board question means role "Director", NOT the word "board" -
        filter on role values, not on the literal phrase.
        Example - the board of a company:
          MATCH (p:Person)-[:{REL_WORKS_FOR}]->(s:Stock {{ticker: 'NVDA'}})
          WHERE toLower(p.role) = 'director'
          RETURN p.name AS director
    """,
    "corporate_structure": f"""
        The company's own corporate family, plus the vendors its SEC filings say
        it depends on. This is filing-derived context, not market data.

        Subsidiaries - the company OWNS them, edge points away from the stock:
          (:{STOCK})-[:{REL_OWNS}]->(:{SUBSIDIARY} {{name: string}})

        Suppliers are POLYMORPHIC: the left side of SUPPLIES_TO may be either a
        tracked company or an unlisted vendor, and BOTH use one relationship type:
          (:{STOCK} supplier)-[:{REL_SUPPLIES_TO}]->(:{STOCK} customer)
          (:{SUPPLIER} vendor)-[:{REL_SUPPLIES_TO}]->(:{STOCK} customer)

        So do NOT put a label on the supplier side - a label on one end silently
        drops half the answer. Leave it unlabeled; an unlabeled node matches both:
          MATCH (sup)-[:{REL_SUPPLIES_TO}]->(s:Stock {{ticker: 'NVDA'}})
          RETURN coalesce(sup.ticker, sup.name) AS supplier

        coalesce is required, not decorative: {SUPPLIER} nodes carry `name` but
        have NO `ticker` property, so `RETURN sup.ticker` yields null for a
        vendor. `coalesce(sup.ticker, sup.name)` gives the ticker for a tracked
        stock and the company name for a vendor.

        Suppliers come from both curated supply-chain data and SEC 10-K
        disclosures; both are real, so do not filter on provenance. Provenance,
        where present, lives on the RELATIONSHIP (r.source, r.critical) and
        never on the node.
        To find who a company supplies, the edge points OUT of it:
          MATCH (s:Stock {{ticker: 'NVDA'}})-[:{REL_SUPPLIES_TO}]->(c:Stock)

        Subsidiaries and untracked suppliers have no price or market-cap
        properties - do not RETURN those.
    """,
    "conversation_memory": f"""
        Node: {QUERY}
        Properties:
          id       (string, unique) - hash of the normalised question
          text     (string)         - the question as the user asked it
          intent   (string)
          count    (integer)        - how many times it has been asked
          first_at (datetime), last_at (datetime)
        Relationship: (:{QUERY})-[:{REL_ABOUT}]->(:{STOCK})

        This is the user's OWN question history - what they asked, and which
        companies each question touched. It is not market data.
        Use it for "what have I asked about", "have I asked this before",
        "which companies do I ask about most", "what did we discuss last".
        Example - companies the user asks about most (`sum` so a repeated
        question counts each time it was asked, not once):
          MATCH (q:{QUERY})-[:{REL_ABOUT}]->(s:{STOCK})
          RETURN s.ticker AS ticker, sum(q.count) AS questions
          ORDER BY questions DESC
    """,
}


def mini_schema(keys: list[str]) -> str:
    """Render the selected subset of the dictionary for a prompt.

    Unknown keys are dropped rather than raising: Step 2's output is
    LLM-generated and already validated against this dictionary, and a stale
    key should degrade to "less context", never to a crash mid-demo.
    """
    blocks = [
        f"### {key}\n{SCHEMA_DICTIONARY[key].strip()}"
        for key in keys
        if key in SCHEMA_DICTIONARY
    ]
    return "\n\n".join(blocks)
