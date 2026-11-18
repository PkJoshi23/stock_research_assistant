"""The evaluation question set and the graph fixture it runs against.

Each entry is one user question plus the assertions we are willing to make about
it. Three kinds of expectation are recorded, because they become true at
different times:

``expected_intent`` / ``expected_schema_keys``
    Product truth: what the agent should do when the language model is working.
    ``expected_schema_keys`` is a *subset* requirement - the router may select
    additional valid keys, but everything listed here must be present. It is
    enforced by ``--live``, and by the offline run whenever the deterministic
    fallback happens to agree.

``reference_cypher`` / ``expected_*``
    A reviewed, read-only query that answers the question against the eval
    fixture, plus the row shape it must return. Offline there is no model, so
    this is how the *data contract* is checked against the real embedded engine:
    labels, relationship direction, the supplier ``coalesce``, the board
    ``role = 'director'`` filter, and the symmetric competitor edge.

``offline_gap``
    Set when the deterministic router fallback cannot produce the product
    intent. Those questions are reported as gaps and are NOT counted as offline
    passes (they are not silently lowered either); ``--live`` is the mode that
    can verify them, because only a model can classify an off-graph question.

The eval store is ``tests/conftest.seed_demo_graph`` (loaded read-only) plus
``EVAL_FIXTURE`` below. The fixture only adds graph shapes the demo seed omits -
subsidiaries, directors, and unlisted ``Supplier`` vendors - so every reference
query has a known, non-empty answer. It is deliberately small and synthetic; see
``evals/README.md`` for exactly what that does and does not prove.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agent.agent_router import (
    INTENT_GENERAL,
    INTENT_GRAPH_QUERY,
    INTENT_HISTORICAL,
    INTENT_LIVE_PRICE,
)
from src.schema import (
    PERSON,
    REL_OWNS,
    REL_SUPPLIES_TO,
    REL_WORKS_FOR,
    STOCK,
    SUBSIDIARY,
    SUPPLIER,
)

#: Extra fixtures written after ``seed_demo_graph``. The seed already provides
#: four Technology stocks, one COMPETES_WITH edge (NVDA->AMD), two Stock
#: SUPPLIES_TO edges (TSM->NVDA, NVDA->MSFT), Jensen Huang as NVDA CEO and four
#: news articles. It has no subsidiary, no director and no unlisted vendor, so
#: those are added here. Parameterised Cypher, same as the ingestion scripts.
EVAL_FIXTURE: list[tuple[str, dict]] = [
    # Subsidiaries: the stock OWNS them.
    (
        f"MERGE (sub:{SUBSIDIARY} {{name: $name}}) "
        f"WITH sub MATCH (s:{STOCK} {{ticker: $ticker}}) "
        f"MERGE (s)-[:{REL_OWNS}]->(sub)",
        {"name": "NVIDIA Graphics", "ticker": "NVDA"},
    ),
    (
        f"MERGE (sub:{SUBSIDIARY} {{name: $name}}) "
        f"WITH sub MATCH (s:{STOCK} {{ticker: $ticker}}) "
        f"MERGE (s)-[:{REL_OWNS}]->(sub)",
        {"name": "Mellanox Technologies", "ticker": "NVDA"},
    ),
    (
        f"MERGE (sub:{SUBSIDIARY} {{name: $name}}) "
        f"WITH sub MATCH (s:{STOCK} {{ticker: $ticker}}) "
        f"MERGE (s)-[:{REL_OWNS}]->(sub)",
        {"name": "LinkedIn", "ticker": "MSFT"},
    ),
    # Directors: Person -> WORKS_FOR -> Stock with role 'Director'.
    (
        f"MERGE (p:{PERSON} {{name: $name}}) SET p.role = $role "
        f"WITH p MATCH (s:{STOCK} {{ticker: $ticker}}) "
        f"MERGE (p)-[:{REL_WORKS_FOR}]->(s)",
        {"name": "Tench Coxe", "role": "Director", "ticker": "NVDA"},
    ),
    (
        f"MERGE (p:{PERSON} {{name: $name}}) SET p.role = $role "
        f"WITH p MATCH (s:{STOCK} {{ticker: $ticker}}) "
        f"MERGE (p)-[:{REL_WORKS_FOR}]->(s)",
        {"name": "A. Brooke Seawell", "role": "Director", "ticker": "NVDA"},
    ),
    (
        f"MERGE (p:{PERSON} {{name: $name}}) SET p.role = $role "
        f"WITH p MATCH (s:{STOCK} {{ticker: $ticker}}) "
        f"MERGE (p)-[:{REL_WORKS_FOR}]->(s)",
        {"name": "Reid Hoffman", "role": "Director", "ticker": "MSFT"},
    ),
    # Unlisted vendors: Supplier node (no ticker) -> SUPPLIES_TO -> Stock. This
    # is the shape that a labelled supplier end would silently drop.
    (
        f"MERGE (v:{SUPPLIER} {{name: $name}}) "
        f"WITH v MATCH (s:{STOCK} {{ticker: $ticker}}) "
        f"MERGE (v)-[:{REL_SUPPLIES_TO}]->(s)",
        {"name": "SK Hynix", "ticker": "NVDA"},
    ),
    (
        f"MERGE (v:{SUPPLIER} {{name: $name}}) "
        f"WITH v MATCH (s:{STOCK} {{ticker: $ticker}}) "
        f"MERGE (v)-[:{REL_SUPPLIES_TO}]->(s)",
        {"name": "Contoso Cloud", "ticker": "MSFT"},
    ),
]

#: The exact vector-search template the schema documents. Offline it is never
#: executed (that would load the embedding model); it is still run through the
#: read-only guard, which is what this entry checks.
_NEWS_VECTOR_TEMPLATE = """CALL db.index.vector.queryNodes('news_embedding_idx', $k, $query_vector)
YIELD node, score
RETURN node.title AS title, node.source AS source, node.published AS published, score
ORDER BY score DESC"""

#: Why the deterministic fallback cannot satisfy the product intent. The router
#: fallback has only three branches (historical / live price / everything else),
#: so an unrelated question is deliberately routed to ``graph_query``. That is a
#: real limitation, recorded here instead of being counted as a pass.
_OFF_GRAPH_GAP = (
    "the deterministic router fallback has no general_knowledge branch; "
    "only a working model can classify an off-graph question"
)


@dataclass(frozen=True)
class Question:
    """One eval question plus every assertion attached to it."""

    id: str
    category: str
    question: str
    expected_intent: str
    expected_schema_keys: frozenset[str] = frozenset()
    #: Reviewed read-only Cypher answering the question against the fixture.
    reference_cypher: str | None = None
    #: False for queries that must not run offline (they need an embedding).
    execute_reference: bool = True
    expected_columns: frozenset[str] = frozenset()
    expected_row_count: int | None = None
    min_rows: int = 0
    #: (column, value) pairs that must appear somewhere in the result.
    expected_cells: tuple[tuple[str, str], ...] = ()
    #: (column, value) pairs that must NOT appear (e.g. a CEO in a board list).
    forbidden_cells: tuple[tuple[str, str], ...] = ()
    #: Included in the ``--live`` routing subset (one real LLM call each).
    live: bool = False
    #: Non-empty when the product intent is not reachable offline.
    offline_gap: str = ""


QUESTIONS: tuple[Question, ...] = (
    # ------------------------------------------------------------------
    # Competitors - stored one-directionally, must be matched undirected.
    # ------------------------------------------------------------------
    Question(
        id="comp_nvda",
        category="competitors",
        question="Which companies compete with Nvidia?",
        expected_intent=INTENT_GRAPH_QUERY,
        expected_schema_keys=frozenset({"competitors"}),
        reference_cypher=(
            "MATCH (s:Stock {ticker: 'NVDA'})-[:COMPETES_WITH]-(c:Stock) "
            "RETURN c.ticker AS ticker, c.name AS name ORDER BY ticker"
        ),
        expected_columns=frozenset({"ticker", "name"}),
        expected_row_count=1,
        expected_cells=(("ticker", "AMD"),),
        live=True,
    ),
    Question(
        id="comp_amd",
        category="competitors",
        question="Who are AMD's competitors?",
        expected_intent=INTENT_GRAPH_QUERY,
        expected_schema_keys=frozenset({"competitors"}),
        reference_cypher=(
            "MATCH (s:Stock {ticker: 'AMD'})-[:COMPETES_WITH]-(c:Stock) "
            "RETURN c.ticker AS ticker ORDER BY ticker"
        ),
        expected_columns=frozenset({"ticker"}),
        expected_row_count=1,
        expected_cells=(("ticker", "NVDA"),),
    ),
    # ------------------------------------------------------------------
    # Suppliers - direction is supplier -> customer and the left end is
    # polymorphic (Stock OR Supplier), so coalesce is the correct return.
    # ------------------------------------------------------------------
    Question(
        id="sup_nvda",
        category="suppliers",
        question="Who supplies Nvidia?",
        expected_intent=INTENT_GRAPH_QUERY,
        expected_schema_keys=frozenset({"supply_chain"}),
        reference_cypher=(
            "MATCH (sup)-[:SUPPLIES_TO]->(s:Stock {ticker: 'NVDA'}) "
            "RETURN coalesce(sup.ticker, sup.name) AS supplier ORDER BY supplier"
        ),
        expected_columns=frozenset({"supplier"}),
        expected_row_count=2,
        expected_cells=(("supplier", "TSM"), ("supplier", "SK Hynix")),
    ),
    Question(
        id="sup_msft",
        category="suppliers",
        question="Which suppliers does Microsoft depend on?",
        expected_intent=INTENT_GRAPH_QUERY,
        expected_schema_keys=frozenset({"supply_chain"}),
        reference_cypher=(
            "MATCH (sup)-[:SUPPLIES_TO]->(s:Stock {ticker: 'MSFT'}) "
            "RETURN coalesce(sup.ticker, sup.name) AS supplier ORDER BY supplier"
        ),
        expected_columns=frozenset({"supplier"}),
        expected_row_count=2,
        expected_cells=(("supplier", "NVDA"), ("supplier", "Contoso Cloud")),
    ),
    Question(
        id="sup_customers_nvda",
        category="suppliers",
        question="Which companies does Nvidia supply to?",
        expected_intent=INTENT_GRAPH_QUERY,
        expected_schema_keys=frozenset({"supply_chain"}),
        reference_cypher=(
            "MATCH (s:Stock {ticker: 'NVDA'})-[:SUPPLIES_TO]->(c:Stock) "
            "RETURN c.ticker AS customer ORDER BY customer"
        ),
        expected_columns=frozenset({"customer"}),
        expected_row_count=1,
        expected_cells=(("customer", "MSFT"),),
    ),
    # ------------------------------------------------------------------
    # Subsidiaries - (Stock)-[:OWNS]->(:Subsidiary).
    # ------------------------------------------------------------------
    Question(
        id="sub_nvda",
        category="subsidiaries",
        question="What subsidiaries does Nvidia own?",
        expected_intent=INTENT_GRAPH_QUERY,
        expected_schema_keys=frozenset({"corporate_structure"}),
        reference_cypher=(
            "MATCH (s:Stock {ticker: 'NVDA'})-[:OWNS]->(sub:Subsidiary) "
            "RETURN sub.name AS subsidiary ORDER BY subsidiary"
        ),
        expected_columns=frozenset({"subsidiary"}),
        expected_row_count=2,
        expected_cells=(
            ("subsidiary", "NVIDIA Graphics"),
            ("subsidiary", "Mellanox Technologies"),
        ),
        live=True,
    ),
    Question(
        id="sub_msft",
        category="subsidiaries",
        question="Which companies has Microsoft acquired?",
        expected_intent=INTENT_GRAPH_QUERY,
        expected_schema_keys=frozenset({"corporate_structure"}),
        reference_cypher=(
            "MATCH (s:Stock {ticker: 'MSFT'})-[:OWNS]->(sub:Subsidiary) "
            "RETURN sub.name AS subsidiary ORDER BY subsidiary"
        ),
        expected_columns=frozenset({"subsidiary"}),
        expected_row_count=1,
        expected_cells=(("subsidiary", "LinkedIn"),),
    ),
    # ------------------------------------------------------------------
    # Board / directors - a board question means role = 'director', NOT the
    # word "board"; the CEO must not be returned in a director list.
    # ------------------------------------------------------------------
    Question(
        id="board_nvda",
        category="board",
        question="Who is on Nvidia's board of directors?",
        expected_intent=INTENT_GRAPH_QUERY,
        expected_schema_keys=frozenset({"executives"}),
        reference_cypher=(
            "MATCH (p:Person)-[:WORKS_FOR]->(s:Stock {ticker: 'NVDA'}) "
            "WHERE toLower(p.role) = 'director' "
            "RETURN p.name AS director ORDER BY director"
        ),
        expected_columns=frozenset({"director"}),
        expected_row_count=2,
        expected_cells=(
            ("director", "Tench Coxe"),
            ("director", "A. Brooke Seawell"),
        ),
        forbidden_cells=(("director", "Jensen Huang"),),
        live=True,
    ),
    Question(
        id="board_msft",
        category="board",
        question="Who are the directors of Microsoft?",
        expected_intent=INTENT_GRAPH_QUERY,
        expected_schema_keys=frozenset({"executives"}),
        reference_cypher=(
            "MATCH (p:Person)-[:WORKS_FOR]->(s:Stock {ticker: 'MSFT'}) "
            "WHERE toLower(p.role) = 'director' "
            "RETURN p.name AS director ORDER BY director"
        ),
        expected_columns=frozenset({"director"}),
        expected_row_count=1,
        expected_cells=(("director", "Reid Hoffman"),),
    ),
    Question(
        id="ceo_nvda",
        category="board",
        question="Who is the CEO of Nvidia?",
        expected_intent=INTENT_GRAPH_QUERY,
        expected_schema_keys=frozenset({"executives"}),
        reference_cypher=(
            "MATCH (p:Person)-[:WORKS_FOR]->(s:Stock {ticker: 'NVDA'}) "
            "WHERE toLower(p.role) = 'ceo' RETURN p.name AS ceo"
        ),
        expected_columns=frozenset({"ceo"}),
        expected_row_count=1,
        expected_cells=(("ceo", "Jensen Huang"),),
    ),
    # ------------------------------------------------------------------
    # Sector - one hop to a Sector node, both directions of the question.
    # ------------------------------------------------------------------
    Question(
        id="sector_nvda",
        category="sector",
        question="What sector is Nvidia in?",
        expected_intent=INTENT_GRAPH_QUERY,
        expected_schema_keys=frozenset({"sector_membership"}),
        reference_cypher=(
            "MATCH (s:Stock {ticker: 'NVDA'})-[:IN_SECTOR]->(sec:Sector) "
            "RETURN sec.name AS sector"
        ),
        expected_columns=frozenset({"sector"}),
        expected_row_count=1,
        expected_cells=(("sector", "Technology"),),
        live=True,
    ),
    Question(
        id="sector_members",
        category="sector",
        question="Which stocks are in the Technology sector?",
        expected_intent=INTENT_GRAPH_QUERY,
        expected_schema_keys=frozenset({"sector_membership"}),
        reference_cypher=(
            "MATCH (s:Stock)-[:IN_SECTOR]->(sec:Sector {name: 'Technology'}) "
            "RETURN s.ticker AS ticker ORDER BY ticker"
        ),
        expected_columns=frozenset({"ticker"}),
        expected_row_count=4,
        expected_cells=(
            ("ticker", "NVDA"),
            ("ticker", "AMD"),
            ("ticker", "TSM"),
            ("ticker", "MSFT"),
        ),
    ),
    Question(
        id="sector_msft",
        category="sector",
        question="What sector does Microsoft belong to?",
        expected_intent=INTENT_GRAPH_QUERY,
        expected_schema_keys=frozenset({"sector_membership"}),
        reference_cypher=(
            "MATCH (s:Stock {ticker: 'MSFT'})-[:IN_SECTOR]->(sec:Sector) "
            "RETURN sec.name AS sector"
        ),
        expected_columns=frozenset({"sector"}),
        expected_row_count=1,
        expected_cells=(("sector", "Technology"),),
    ),
    # ------------------------------------------------------------------
    # News - per-stock mention traversal, and the semantic template.
    # ------------------------------------------------------------------
    Question(
        id="news_nvda",
        category="news",
        question="What recent news mentions Nvidia?",
        expected_intent=INTENT_GRAPH_QUERY,
        expected_schema_keys=frozenset({"news_by_stock"}),
        reference_cypher=(
            "MATCH (s:Stock {ticker: 'NVDA'})-[:MENTIONED_IN]->(n:NewsArticle) "
            "RETURN n.title AS title, n.source AS source "
            "ORDER BY n.published DESC LIMIT 5"
        ),
        expected_columns=frozenset({"title", "source"}),
        expected_row_count=3,
        expected_cells=(("title", "Nvidia beats earnings"),),
        live=True,
    ),
    Question(
        id="news_amd",
        category="news",
        question="Show me news about AMD",
        expected_intent=INTENT_GRAPH_QUERY,
        expected_schema_keys=frozenset({"news_by_stock"}),
        reference_cypher=(
            "MATCH (s:Stock {ticker: 'AMD'})-[:MENTIONED_IN]->(n:NewsArticle) "
            "RETURN n.title AS title, n.source AS source "
            "ORDER BY n.published DESC LIMIT 5"
        ),
        expected_columns=frozenset({"title", "source"}),
        expected_row_count=1,
        expected_cells=(("title", "AMD launches new chip"),),
    ),
    Question(
        id="news_semantic",
        category="news",
        question="What is the market saying about AI chips?",
        expected_intent=INTENT_GRAPH_QUERY,
        expected_schema_keys=frozenset({"news_semantic_search"}),
        reference_cypher=_NEWS_VECTOR_TEMPLATE,
        # Executing this would load the embedding model; offline only the
        # read-only guard is checked. Reaching the vector index is a live /
        # ingestion concern, not an offline one.
        execute_reference=False,
        live=True,
    ),
    # ------------------------------------------------------------------
    # Live price - answered by yfinance, not the graph. Offline only the
    # routing decision and the generator's "skip" contract are checked.
    # ------------------------------------------------------------------
    Question(
        id="live_nvda",
        category="live_price",
        question="What is NVDA trading at right now?",
        expected_intent=INTENT_LIVE_PRICE,
        live=True,
    ),
    Question(
        id="live_msft",
        category="live_price",
        question="Current price of Microsoft please",
        expected_intent=INTENT_LIVE_PRICE,
    ),
    Question(
        id="live_aapl",
        category="live_price",
        question="Is Apple worth buying today?",
        expected_intent=INTENT_LIVE_PRICE,
    ),
    # ------------------------------------------------------------------
    # Historical trend - EXPLICITLY DEFERRED. The graph stores no price
    # history, so these must never become graph queries.
    # ------------------------------------------------------------------
    Question(
        id="hist_tsla",
        category="historical",
        question="How has Tesla's stock performed since 2020?",
        expected_intent=INTENT_HISTORICAL,
        live=True,
    ),
    Question(
        id="hist_chart",
        category="historical",
        question="Show me a 5-year chart of Apple",
        expected_intent=INTENT_HISTORICAL,
    ),
    Question(
        id="hist_last_month",
        category="historical",
        question="What was Microsoft's price last month?",
        expected_intent=INTENT_HISTORICAL,
    ),
    Question(
        id="hist_moving_average",
        category="historical",
        question="Give me the moving average of Intel",
        expected_intent=INTENT_HISTORICAL,
    ),
    Question(
        id="hist_trended",
        category="historical",
        question="How has NVDA trended over the past year?",
        expected_intent=INTENT_HISTORICAL,
    ),
    # ------------------------------------------------------------------
    # Unrelated questions - product intent is general_knowledge. The offline
    # fallback cannot reach it, so these are tracked as gaps, never passes.
    # ------------------------------------------------------------------
    Question(
        id="gen_pe_ratio",
        category="general",
        question="What is a P/E ratio?",
        expected_intent=INTENT_GENERAL,
        offline_gap=_OFF_GRAPH_GAP,
        live=True,
    ),
    Question(
        id="gen_world_cup",
        category="general",
        question="Who won the world cup?",
        expected_intent=INTENT_GENERAL,
        offline_gap=_OFF_GRAPH_GAP,
    ),
    Question(
        id="gen_capital_gains",
        category="general",
        question="What is a capital gains tax?",
        expected_intent=INTENT_GENERAL,
        offline_gap=_OFF_GRAPH_GAP,
    ),
)
