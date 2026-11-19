"""Subgraph extraction and pyvis rendering.

The bugs these tests pin down, all seen in the first implementation:
  * nodes without a label fell back to "Unknown" and collapsed into one node;
  * ids were bare strings, so a sector and a company could collide;
  * edge orientation was lost, so a supplier looked like a customer.
"""

from __future__ import annotations

import json
import re

from src import visualization
from src.schema import PERSON, REL_SUPPLIES_TO, REL_OWNS, SECTOR, STOCK, SUBSIDIARY


def _kinds(subgraph: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in subgraph["nodes"]:
        kind = node["id"].split(":", 1)[0]
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def test_subgraph_has_the_expected_shape(seeded):
    subgraph = visualization.fetch_subgraph(seeded, ["NVDA"])
    # NVDA + AMD (competitor) + TSM (supplier) + MSFT (customer), plus the
    # sector, one person and one subsidiary written by the seed graph.
    assert _kinds(subgraph) == {
        STOCK: 4,
        SECTOR: 1,
        PERSON: 1,
        SUBSIDIARY: 1,
    }


def test_every_node_has_a_namespaced_unique_id(seeded):
    subgraph = visualization.fetch_subgraph(seeded, ["NVDA", "AMD"])
    ids = [node["id"] for node in subgraph["nodes"]]
    assert len(ids) == len(set(ids))
    assert all(":" in node_id for node_id in ids)


def test_supply_edge_keeps_its_direction(seeded):
    subgraph = visualization.fetch_subgraph(seeded, ["NVDA"])
    supply_edges = [edge for edge in subgraph["edges"] if edge["rel"] == REL_SUPPLIES_TO]
    pairs = {(edge["from"], edge["to"]) for edge in supply_edges}
    assert (f"{STOCK}:TSM", f"{STOCK}:NVDA") in pairs
    assert (f"{STOCK}:NVDA", f"{STOCK}:MSFT") in pairs
    # and never reversed
    assert (f"{STOCK}:NVDA", f"{STOCK}:TSM") not in pairs


def test_unknown_ticker_returns_nothing(seeded):
    assert visualization.fetch_subgraph(seeded, ["NOPE"]) == {"nodes": [], "edges": []}


def test_empty_ticker_list_returns_nothing(seeded):
    assert visualization.fetch_subgraph(seeded, []) == {"nodes": [], "edges": []}


def test_edges_only_reference_existing_nodes(seeded):
    subgraph = visualization.fetch_subgraph(seeded, ["NVDA"])
    ids = {node["id"] for node in subgraph["nodes"]}
    for edge in subgraph["edges"]:
        assert edge["from"] in ids and edge["to"] in ids


def test_html_embeds_the_network_data(seeded):
    html = visualization.build_pyvis_html(
        visualization.fetch_subgraph(seeded, ["NVDA"])
    )
    assert "vis.DataSet" in html
    match = re.search(r"nodes\s*=\s*new vis\.DataSet\((\[.*?\])\);", html, re.S)
    assert match, "node dataset not found in generated HTML"
    data = json.loads(match.group(1))
    assert len(data) >= 4
    assert any(node["label"] == "NVDA" for node in data)


def test_html_is_self_contained(seeded):
    """Streamlit serves the document from an iframe: no relative asset paths."""
    html = visualization.build_pyvis_html(visualization.fetch_subgraph(seeded, ["NVDA"]))
    assert "vis-network" in html
    assert 'src="./lib/' not in html


def test_options_are_valid_json(seeded):
    """set_options parses JSON; a malformed object fails at render time."""
    html = visualization.build_pyvis_html(visualization.fetch_subgraph(seeded, ["NVDA"]))
    match = re.search(r"var options = (\{.*?\});", html, re.S)
    assert match
    options = json.loads(match.group(1))
    assert options["physics"]["enabled"] is True
