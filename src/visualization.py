"""Subgraph extraction and pyvis rendering.

The graph is the product, so the demo needs to *show* it. This module turns a
set of seed tickers into a self-contained interactive HTML document:

    fetch_subgraph(session, ["NVDA"])  ->  {"nodes": [...], "edges": [...]}
    build_pyvis_html(subgraph)         ->  "<html>...vis.js...</html>"

Two deliberate choices:

* **Property-only results.** Every query returns scalars (`s.ticker`, `n.title`,
  ...) instead of whole nodes. Relate hands back native `Node` objects for
  `RETURN s`, which do not survive a trip through JSON or Streamlit, and the
  previous implementation worked around that by falling back to the string
  "Unknown" - which collapsed every news article into one node.
* **Namespaced node ids.** Ids are ``Stock:NVDA`` / ``Sector:Technology`` /
  ``News:<url>``, so a company, a sector and a headline can never collide.
"""

from __future__ import annotations

import json
from typing import Any

from .schema import (
    NEWS,
    PERSON,
    REL_COMPETES_WITH,
    REL_IN_SECTOR,
    REL_MENTIONED_IN,
    REL_OWNS,
    REL_SUPPLIES_TO,
    REL_WORKS_FOR,
    SECTOR,
    STOCK,
    SUBSIDIARY,
    SUPPLIER,
)

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

#: One style per label. Colours are chosen to stay distinguishable on the dark
#: Streamlit background, and shapes differ too - colour alone fails for
#: colour-blind judges, and this is a demo that will be projected.
NODE_STYLES: dict[str, dict[str, Any]] = {
    STOCK: {"color": "#3498db", "size": 26, "shape": "dot"},
    SECTOR: {"color": "#2ecc71", "size": 18, "shape": "database"},
    PERSON: {"color": "#f39c12", "size": 16, "shape": "triangle"},
    NEWS: {"color": "#9b59b6", "size": 9, "shape": "square"},
    # Filing-derived nodes are drawn smaller and darker: they are context, not
    # the subject of the answer.
    SUBSIDIARY: {"color": "#16a085", "size": 12, "shape": "box"},
    SUPPLIER: {"color": "#d4ac0d", "size": 12, "shape": "hexagon"},
}

#: Edge colour + dash per relationship type, matching the legend in the UI.
EDGE_STYLES: dict[str, dict[str, Any]] = {
    REL_IN_SECTOR: {"color": "#2ecc71", "dashes": True, "label": "in sector"},
    REL_COMPETES_WITH: {"color": "#e74c3c", "dashes": [2, 4], "label": "competes"},
    REL_SUPPLIES_TO: {"color": "#f1c40f", "dashes": False, "label": "supplies"},
    REL_WORKS_FOR: {"color": "#f39c12", "dashes": False, "label": "works for"},
    REL_MENTIONED_IN: {"color": "#8e44ad", "dashes": [1, 3], "label": "news"},
    REL_OWNS: {"color": "#16a085", "dashes": [4, 4], "label": "owns"},
}


def node_id(label: str, key: str) -> str:
    """Namespaced id, so 'Technology' the sector never equals a stock's name."""
    return f"{label}:{key}"


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------


def _fmt_money(value: float | None) -> str:
    if not value:
        return "n/a"
    if value >= 1e12:
        return f"${value / 1e12:,.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:,.2f}B"
    return f"${value:,.2f}"


def fetch_subgraph(
    session,
    tickers: list[str],
    *,
    news_per_stock: int = 3,
    include_news: bool = True,
) -> dict[str, list[dict]]:
    """Collect the neighbourhood of ``tickers`` as plain nodes and edges.

    Depth is one hop, by design: two hops across a graph with 60+ news nodes
    produces a hairball that says nothing. The one-hop view answers "what is
    this company connected to?" which is the question the UI is making.
    """
    if not tickers:
        return {"nodes": [], "edges": []}

    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str, str], dict] = {}

    def add_node(kind: str, key: str, **props) -> str:
        # ``kind``/``key`` rather than label/key: callers pass ``label=`` for the
        # display text (e.g. a truncated headline), which would otherwise collide
        # with a positional parameter of the same name.
        nid = node_id(kind, key)
        if nid not in nodes:
            nodes[nid] = {
                "id": nid,
                "label": key,
                **NODE_STYLES.get(kind, {}),
                **props,
            }
        return nid

    def add_edge(source: str, target: str, rel: str) -> None:
        key = (source, target, rel)
        if key not in edges:
            edges[key] = {"from": source, "to": target, "rel": rel}

    # -- stocks ---------------------------------------------------------
    for row in session.run(
        f"""
        MATCH (s:{STOCK})
        WHERE s.ticker IN $tickers
        RETURN s.ticker AS ticker, s.name AS name, s.sector AS sector,
               s.current_price AS price, s.market_cap AS cap, s.pe_ratio AS pe
        """,
        {"tickers": tickers},
    ):
        tooltip = (
            f"<b>{row['ticker']}</b><br>{row['name'] or ''}<br>"
            f"Sector: {row['sector'] or 'n/a'}<br>"
            f"Price: {_fmt_money(row['price'])}<br>"
            f"Market cap: {_fmt_money(row['cap'])}<br>"
            f"P/E: {(row['pe'] if row['pe'] else 0):.1f}"
        )
        add_node(STOCK, row["ticker"], title=tooltip, group=STOCK)

    # -- sectors --------------------------------------------------------
    for row in session.run(
        f"""
        MATCH (s:{STOCK})-[:{REL_IN_SECTOR}]->(sec:{SECTOR})
        WHERE s.ticker IN $tickers
        RETURN DISTINCT s.ticker AS ticker, sec.name AS sector
        """,
        {"tickers": tickers},
    ):
        if not row["sector"]:
            continue
        source = node_id(STOCK, row["ticker"])
        target = add_node(
            SECTOR, row["sector"], title=f"<b>{row['sector']}</b><br>Sector", group=SECTOR
        )
        add_edge(source, target, REL_IN_SECTOR)

    # -- competitors (stored one-way, drawn both ways) ------------------
    for row in session.run(
        f"""
        MATCH (s:{STOCK})-[:{REL_COMPETES_WITH}]-(other:{STOCK})
        WHERE s.ticker IN $tickers
        RETURN DISTINCT s.ticker AS ticker, other.ticker AS other,
                        other.name AS other_name
        """,
        {"tickers": tickers},
    ):
        source = add_node(STOCK, row["ticker"])
        target = add_node(
            STOCK,
            row["other"],
            title=f"<b>{row['other']}</b><br>{row['other_name'] or ''}",
            group=STOCK,
        )
        add_edge(source, target, REL_COMPETES_WITH)

    # -- supply chain, both directions, with the real orientation -------
    for row in session.run(
        f"""
        MATCH (s:{STOCK})-[:{REL_SUPPLIES_TO}]->(c:{STOCK})
        WHERE s.ticker IN $tickers
        RETURN DISTINCT s.ticker AS supplier, c.ticker AS customer, c.name AS name
        """,
        {"tickers": tickers},
    ):
        add_node(STOCK, row["supplier"])
        add_node(STOCK, row["customer"], title=f"<b>{row['customer']}</b><br>{row['name'] or ''}", group=STOCK)
        add_edge(node_id(STOCK, row["supplier"]), node_id(STOCK, row["customer"]), REL_SUPPLIES_TO)

    for row in session.run(
        f"""
        MATCH (sup:{STOCK})-[:{REL_SUPPLIES_TO}]->(s:{STOCK})
        WHERE s.ticker IN $tickers
        RETURN DISTINCT sup.ticker AS supplier, s.ticker AS customer, sup.name AS name
        """,
        {"tickers": tickers},
    ):
        add_node(STOCK, row["customer"])
        add_node(STOCK, row["supplier"], title=f"<b>{row['supplier']}</b><br>{row['name'] or ''}", group=STOCK)
        add_edge(node_id(STOCK, row["supplier"]), node_id(STOCK, row["customer"]), REL_SUPPLIES_TO)

    # -- executives -----------------------------------------------------
    for row in session.run(
        f"""
        MATCH (p:{PERSON})-[:{REL_WORKS_FOR}]->(s:{STOCK})
        WHERE s.ticker IN $tickers
        RETURN DISTINCT p.name AS name, p.role AS role, s.ticker AS ticker
        """,
        {"tickers": tickers},
    ):
        target = add_node(STOCK, row["ticker"])
        source = add_node(
            PERSON,
            row["name"],
            title=f"<b>{row['name']}</b><br>{row['role'] or 'Executive'}",
            group=PERSON,
        )
        add_edge(source, target, REL_WORKS_FOR)

    # -- subsidiaries (SEC 10-K) ----------------------------------------
    for row in session.run(
        f"""
        MATCH (s:{STOCK})-[:{REL_OWNS}]->(sub:{SUBSIDIARY})
        WHERE s.ticker IN $tickers
        RETURN DISTINCT s.ticker AS ticker, sub.name AS name
        """,
        {"tickers": tickers},
    ):
        target = add_node(
            SUBSIDIARY,
            row["name"],
            title=f"<b>{row['name']}</b><br>Subsidiary of {row['ticker']} (10-K)",
            group=SUBSIDIARY,
        )
        add_edge(node_id(STOCK, row["ticker"]), target, REL_OWNS)

    # -- named suppliers that are not tracked stocks --------------------
    # Both kinds of supplier share SUPPLIES_TO, so they render identically; the
    # node label is what tells a vendor apart from a listed company.
    for row in session.run(
        f"""
        MATCH (v:{SUPPLIER})-[:{REL_SUPPLIES_TO}]->(s:{STOCK})
        WHERE s.ticker IN $tickers
        RETURN DISTINCT s.ticker AS ticker, v.name AS name
        """,
        {"tickers": tickers},
    ):
        source = add_node(
            SUPPLIER,
            row["name"],
            title=f"<b>{row['name']}</b><br>Named supplier of {row['ticker']} (10-K)",
            group=SUPPLIER,
        )
        add_edge(source, node_id(STOCK, row["ticker"]), REL_SUPPLIES_TO)

    # -- news (bounded per stock) ---------------------------------------
    if include_news and news_per_stock > 0:
        rows = session.run(
            f"""
            MATCH (s:{STOCK})-[:{REL_MENTIONED_IN}]->(n:{NEWS})
            WHERE s.ticker IN $tickers
            RETURN s.ticker AS ticker, n.url AS url, n.title AS title,
                   n.source AS source, n.published AS published
            ORDER BY n.published DESC
            """,
            {"tickers": tickers},
        )
        # Cap per stock: 60 headlines would bury every other relationship.
        per_stock: dict[str, int] = {}
        for row in rows:
            ticker = row["ticker"]
            if per_stock.get(ticker, 0) >= news_per_stock:
                continue
            per_stock[ticker] = per_stock.get(ticker, 0) + 1
            title = row["title"] or "Untitled"
            short = title if len(title) <= 34 else title[:32] + "..."
            source = add_node(
                NEWS,
                row["url"],
                label=short,
                title=f"<b>{title}</b><br>{row['source'] or ''}<br>{row['published'] or ''}",
                group=NEWS,
            )
            add_edge(node_id(STOCK, ticker), source, REL_MENTIONED_IN)

    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def build_pyvis_html(
    subgraph: dict[str, list[dict]],
    *,
    height: str = "560px",
    physics: bool = True,
) -> str:
    """Render nodes/edges as a self-contained pyvis document.

    ``cdn_resources="in_line"`` embeds vis-network inside the document. The
    default (`local`) emits relative ``./lib/vis-*.js`` paths, which 404 inside
    a Streamlit iframe. The cost is a ~1 MB HTML string, which is fine for an
    iframe that is only rebuilt when a question is answered.
    """
    from pyvis.network import Network

    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#0e1117",
        font_color="#fafafa",
        cdn_resources="in_line",
    )

    for node in subgraph["nodes"]:
        net.add_node(
            node["id"],
            label=node.get("label", node["id"].split(":", 1)[-1]),
            title=node.get("title", ""),
            color=node.get("color", "#97c2fc"),
            size=node.get("size", 12),
            shape=node.get("shape", "dot"),
            borderWidth=2,
            borderWidthSelected=4,
            font={"size": 13, "face": "Inter, Segoe UI, sans-serif"},
        )

    for edge in subgraph["edges"]:
        style = EDGE_STYLES.get(edge["rel"], {"color": "#888", "dashes": False, "label": edge["rel"]})
        net.add_edge(
            edge["from"],
            edge["to"],
            title=style["label"],
            color=style["color"],
            dashes=style["dashes"],
            width=1.4,
            arrows="to",
            smooth={"type": "continuous"},
        )

    options = {
        "physics": {
            "enabled": physics,
            "barnesHut": {
                "gravitationalConstant": -11000,
                "centralGravity": 0.25,
                "springLength": 190,
                "springConstant": 0.03,
                "damping": 0.9,
            },
            "stabilization": {"iterations": 220, "updateInterval": 25},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 120,
            "navigationButtons": True,
            "keyboard": {"enabled": False},
            "multiselect": False,
        },
        "edges": {"font": {"size": 11, "color": "#c9d1d9", "strokeWidth": 0}},
    }
    net.set_options(json.dumps(options))
    return net.generate_html(notebook=False)
