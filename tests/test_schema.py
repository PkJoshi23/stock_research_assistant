"""The schema dictionary is a contract, not documentation.

These tests exist because the original code drifted: the ingestion wrote
``Stock`` while the descriptions told the model about ``Company``, so generated
queries were valid Cypher that silently matched nothing.
"""

from __future__ import annotations

import pytest

from src import schema
from src.schema import SCHEMA_DICTIONARY, mini_schema


def test_expected_keys_are_present():
    assert set(SCHEMA_DICTIONARY) == {
        "stock_snapshot",
        "sector_membership",
        "competitors",
        "supply_chain",
        "executives",
        "corporate_structure",
        "conversation_memory",
        "news_by_stock",
        "news_semantic_search",
    }


def test_conversation_memory_key_describes_the_query_node():
    description = SCHEMA_DICTIONARY["conversation_memory"]
    for token in (schema.QUERY, schema.REL_ABOUT, "count", "questions"):
        assert token in description


@pytest.mark.parametrize("key", list(SCHEMA_DICTIONARY))
def test_descriptions_do_not_use_the_abandoned_company_label(key: str):
    """Regression: the loader writes ``Stock``; ``Company`` must not be in any
    prompt, or the model generates queries against a label that never exists."""
    assert "Company" not in SCHEMA_DICTIONARY[key]


def test_descriptions_reference_generated_relationship_constants():
    """Every relationship name mentioned must come from this module."""
    joined = " ".join(SCHEMA_DICTIONARY.values())
    for rel in (
        schema.REL_IN_SECTOR,
        schema.REL_COMPETES_WITH,
        schema.REL_SUPPLIES_TO,
        schema.REL_WORKS_FOR,
        schema.REL_MENTIONED_IN,
    ):
        assert rel in joined


def test_snapshot_description_lists_every_property_the_loader_writes():
    """P/E was written by ingestion but missing from the schema, so the agent
    could never answer a valuation question."""
    description = SCHEMA_DICTIONARY["stock_snapshot"]
    for prop in ("ticker", "name", "aliases", "sector", "industry",
                 "market_cap", "pe_ratio", "current_price", "updated_at"):
        assert prop in description


def test_vector_search_description_uses_the_injected_placeholders():
    """The executor fills $query_vector/$k; a literal would break the CALL."""
    description = SCHEMA_DICTIONARY["news_semantic_search"]
    assert "$query_vector" in description
    assert "$k" in description
    assert schema.VECTOR_INDEX_NAME in description
    assert "db.index.vector.queryNodes" in description


def test_mini_schema_is_empty_for_no_keys():
    assert mini_schema([]) == ""


def test_mini_schema_renders_only_requested_keys():
    rendered = mini_schema(["stock_snapshot"])
    assert "ticker" in rendered
    # A key that was not selected must not leak into the prompt.
    assert "MENTIONED_IN" not in rendered


def test_mini_schema_drops_unknown_keys_instead_of_raising():
    rendered = mini_schema(["stock_snapshot", "not_a_real_key"])
    assert "stock_snapshot" in rendered
    assert "not_a_real_key" not in rendered


def test_node_styles_cover_every_label():
    """The visualisation styles a node per label; a missing label renders grey."""
    from src.visualization import NODE_STYLES

    for label in (
        schema.STOCK,
        schema.SECTOR,
        schema.PERSON,
        schema.NEWS,
        schema.SUBSIDIARY,
        schema.SUPPLIER,
    ):
        assert label in NODE_STYLES


def test_edge_styles_cover_every_relationship():
    from src.visualization import EDGE_STYLES

    for rel in (
        schema.REL_IN_SECTOR,
        schema.REL_COMPETES_WITH,
        schema.REL_SUPPLIES_TO,
        schema.REL_WORKS_FOR,
        schema.REL_MENTIONED_IN,
        schema.REL_OWNS,
    ):
        assert rel in EDGE_STYLES


def test_corporate_structure_describes_the_sec_labels():
    """The SEC loader writes these; the prompt must describe them or the agent
    can never answer a subsidiary/supplier question."""
    description = SCHEMA_DICTIONARY["corporate_structure"]
    for token in (schema.SUBSIDIARY, schema.SUPPLIER, schema.REL_OWNS,
                  schema.REL_SUPPLIES_TO):
        assert token in description


def test_corporate_structure_states_supplier_polymorphism():
    """The model must know the supplier end can be a Stock OR a Supplier, and
    that `coalesce` is required because vendors have no ticker."""
    description = SCHEMA_DICTIONARY["corporate_structure"]
    assert "POLYMORPHIC" in description
    assert "coalesce(sup.ticker, sup.name)" in description
    assert "NO `ticker` property" in description or "NO ticker" in description


def test_suppliers_use_a_single_relationship_type():
    """Both tracked stocks and untracked vendors use SUPPLIES_TO.

    The earlier two-type split forced every supplier query into a two-branch
    union, and the model reliably answered with only one branch.
    """
    description = SCHEMA_DICTIONARY["corporate_structure"]
    assert "CRITICAL_SUPPLIER_FOR" not in description
    assert "UNION" not in description
