"""Few-shot grounding for Step 3 (Cypher generation).

The supplier question failed for three iterations of prose-only rules: the model
kept writing `(sup:Stock)-[:SUPPLIES_TO]->…`, which silently drops the unlisted
vendors, and `sup.ticker` is null for those nodes. What fixed it was a worked
example it could copy. These tests keep that example wired to the right schema
key and out of prompts that did not select it.
"""

from __future__ import annotations

import pytest

from src.agent import cypher_generator as gen


def test_examples_are_gated_by_the_selected_keys():
    """An example must never teach a label the mini-schema omits."""
    unrelated_only = gen._render_examples(["competitors"])
    assert "SUPPLIES_TO" not in unrelated_only
    assert "OWNS" not in unrelated_only

    supplier = gen._render_examples(["corporate_structure"])
    assert "SUPPLIES_TO" in supplier
    assert "OWNS" in supplier


def test_supplier_example_uses_the_unlabeled_polymorphic_pattern():
    """The example is the fix: unlabeled supplier node + coalesce."""
    rendered = gen._render_examples(["corporate_structure"])
    assert "MATCH (sup)-[:SUPPLIES_TO]->(s:Stock" in rendered
    assert "coalesce(sup.ticker, sup.name)" in rendered
    # A labeled supplier end must not appear - that is the regression.
    assert "(sup:Stock)" not in rendered


def test_examples_are_capped():
    rendered = gen._render_examples(
        ["corporate_structure", "executives", "competitors", "supply_chain"]
    )
    assert rendered.count("Cypher:") == gen._MAX_EXAMPLES


def test_no_keys_renders_a_placeholder_not_a_crash():
    assert gen._render_examples([]).startswith("(no example")


def test_prompt_includes_the_supplier_example_when_selected(monkeypatch):
    captured: dict = {}

    def fake_chat(prompt, **kwargs):
        captured["system"] = kwargs.get("system", "")
        return "MATCH (sup)-[:SUPPLIES_TO]->(s:Stock {ticker: 'NVDA'}) RETURN coalesce(sup.ticker, sup.name) AS supplier"

    monkeypatch.setattr(gen.llm, "chat", fake_chat)

    result = gen.generate_cypher_query(
        "who supplies nvidia", ["corporate_structure"], "graph_query"
    )

    assert result["action"] == "graph_query"
    assert "coalesce(sup.ticker, sup.name)" in captured["system"]
    assert "RETURN coalesce(sup.ticker, sup.name) AS supplier" in captured["system"]


def test_prompt_omits_the_supplier_example_when_not_selected(monkeypatch):
    captured: dict = {}

    def fake_chat(prompt, **kwargs):
        captured["system"] = kwargs.get("system", "")
        return "MATCH (s:Stock) RETURN s.ticker AS ticker"

    monkeypatch.setattr(gen.llm, "chat", fake_chat)

    gen.generate_cypher_query("who supplies nvidia", ["stock_snapshot"], "graph_query")

    assert "coalesce(sup.ticker, sup.name)" not in captured["system"]


@pytest.mark.parametrize("intent", ["live_price_fetch", "historical_trend"])
def test_non_graph_intents_never_prompt(intent):
    assert gen.generate_cypher_query("q", ["stock_snapshot"], intent)["action"] == "skip"
