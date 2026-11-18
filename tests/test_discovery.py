"""The crawl engine, driven by fakes.

No network, no LLM, no SEC: the engine takes its extractor and resolver as
callables precisely so the fan-out, the budget and the cycle detection can be
tested deterministically.
"""

from __future__ import annotations

import pytest

from src import discovery


def _resolver(mapping: dict[str, str | None]):
    return lambda name: mapping.get(name.strip().lower())


def _extractor(mapping: dict[str, list[str]], calls: list[str] | None = None):
    def extract(ticker: str) -> list[str]:
        if calls is not None:
            calls.append(ticker)
        return list(mapping.get(ticker, []))

    return extract


def test_single_hop_finds_suppliers():
    result = discovery.discover(
        seeds=["A"],
        extract_suppliers=_extractor({"A": ["B Corp", "C Corp"]}),
        resolve_ticker=_resolver({"b corp": "B", "c corp": "C"}),
        max_depth=1,
    )
    assert [node.ticker for node in result.nodes] == ["A", "B", "C"]
    assert result.edges == [("B", "A"), ("C", "A")]


def test_two_hops_expand_discovered_suppliers():
    result = discovery.discover(
        seeds=["A"],
        extract_suppliers=_extractor({"A": ["B Corp"], "B": ["C Corp"]}),
        resolve_ticker=_resolver({"b corp": "B", "c corp": "C"}),
        max_depth=2,
    )
    assert result.edges == [("B", "A"), ("C", "B")]
    assert result.expanded == 2


def test_depth_limit_stops_expansion():
    calls: list[str] = []
    discovery.discover(
        seeds=["A"],
        extract_suppliers=_extractor({"A": ["B Corp"], "B": ["C Corp"]}, calls),
        resolve_ticker=_resolver({"b corp": "B", "c corp": "C"}),
        max_depth=1,
    )
    assert calls == ["A"]  # B is discovered but never read


def test_unresolved_vendor_is_recorded_not_guessed():
    result = discovery.discover(
        seeds=["A"],
        extract_suppliers=_extractor({"A": ["Some Private Foundry"]}),
        resolve_ticker=_resolver({"some private foundry": None}),
        max_depth=2,
    )
    assert result.unresolved == [("A", "Some Private Foundry")]
    assert [node.ticker for node in result.nodes] == ["A"]


def test_self_reference_is_ignored():
    result = discovery.discover(
        seeds=["A"],
        extract_suppliers=_extractor({"A": ["A Corp"]}),
        resolve_ticker=_resolver({"a corp": "A"}),
        max_depth=2,
    )
    assert result.edges == []


def test_duplicate_edges_are_collapsed():
    result = discovery.discover(
        seeds=["A", "B"],
        extract_suppliers=_extractor({"A": ["B Corp"], "B": ["A Corp"]}),
        resolve_ticker=_resolver({"b corp": "B", "a corp": "A"}),
        max_depth=2,
    )
    assert sorted(result.edges) == [("A", "B"), ("B", "A")]


def test_node_budget_stops_the_crawl_cleanly():
    result = discovery.discover(
        seeds=["A"],
        extract_suppliers=_extractor({"A": ["B Corp", "C Corp"]}),
        resolve_ticker=_resolver({"b corp": "B", "c corp": "C"}),
        max_depth=2,
        max_nodes=2,
    )
    assert [node.ticker for node in result.nodes] == ["A", "B"]
    assert result.exhausted is True
    assert result.deferred  # C was seen but not admitted


def test_cyclic_chain_is_reported():
    result = discovery.discover(
        seeds=["A"],
        extract_suppliers=_extractor({"A": ["B Corp"], "B": ["C Corp"], "C": ["A Corp"]}),
        resolve_ticker=_resolver({"b corp": "B", "c corp": "C", "a corp": "A"}),
        max_depth=3,
    )
    assert result.cycles, "expected a cycle"
    cycle = result.cycles[0]
    assert set(cycle) == {"A", "B", "C"}


def test_no_cycle_for_a_chain():
    result = discovery.discover(
        seeds=["A"],
        extract_suppliers=_extractor({"A": ["B Corp"], "B": ["C Corp"]}),
        resolve_ticker=_resolver({"b corp": "B", "c corp": "C"}),
        max_depth=3,
    )
    assert result.cycles == []


def test_ensure_company_veto_skips_expansion():
    calls: list[str] = []
    result = discovery.discover(
        seeds=["A"],
        extract_suppliers=_extractor({"A": ["B Corp"], "B": ["C Corp"]}, calls),
        resolve_ticker=_resolver({"b corp": "B", "c corp": "C"}),
        ensure_company=lambda ticker: ticker != "B",
        max_depth=2,
    )
    assert calls == ["A"]
    assert any("could not be ingested" in error for error in result.errors)


def test_extraction_failure_does_not_stop_the_crawl():
    def extract(ticker: str) -> list[str]:
        if ticker == "A":
            raise RuntimeError("SEC down")
        return []

    result = discovery.discover(
        seeds=["A", "Z"],
        extract_suppliers=extract,
        resolve_ticker=_resolver({}),
        max_depth=1,
    )
    assert [node.ticker for node in result.nodes] == ["A", "Z"]
    assert any("extraction failed" in error for error in result.errors)


def test_crawl_is_deterministic():
    kwargs = dict(
        seeds=["A", "B"],
        extract_suppliers=_extractor({"A": ["C Corp", "D Corp"], "B": ["D Corp"]}),
        resolve_ticker=_resolver({"c corp": "C", "d corp": "D"}),
        max_depth=2,
    )
    first, second = discovery.discover(**kwargs), discovery.discover(**kwargs)
    assert [(n.ticker, n.hop) for n in first.nodes] == [(n.ticker, n.hop) for n in second.nodes]
    assert first.edges == second.edges


def test_seeds_are_normalised_and_deduplicated():
    result = discovery.discover(
        seeds=[" a ", "A", "b"],
        extract_suppliers=_extractor({}),
        resolve_ticker=_resolver({}),
        max_depth=1,
    )
    assert result.seeds == ["A", "B"]


def test_tree_renders_parent_relationships():
    result = discovery.discover(
        seeds=["A"],
        extract_suppliers=_extractor({"A": ["B Corp"]}),
        resolve_ticker=_resolver({"b corp": "B"}),
        max_depth=1,
    )
    tree = discovery.format_tree(result)
    assert tree[0].startswith("A")
    assert any("B" in line and "└─" in line for line in tree)


@pytest.mark.parametrize(
    "edges,expected",
    [
        ([("A", "B"), ("B", "A")], [["A", "B"]]),
        ([("A", "B"), ("B", "C"), ("C", "A")], [["A", "B", "C"]]),
        ([("A", "B"), ("B", "C")], []),
    ],
)
def test_find_cycles(edges, expected):
    cycles = discovery.find_cycles(edges)
    assert sorted([sorted(cycle) for cycle in cycles]) == sorted([sorted(c) for c in expected])


def test_cycles_are_canonicalised_so_rotations_do_not_duplicate():
    # Same cycle reached from three starting points must appear once.
    cycles = discovery.find_cycles([("A", "B"), ("B", "C"), ("C", "A")])
    assert len(cycles) == 1
    assert cycles[0][0] == min(cycles[0])
