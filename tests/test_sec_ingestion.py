"""SEC ingestion: section selection, LLM-output hardening, graph writes.

No network and no LLM: filings are synthesised, and ``llm.chat_json`` is
patched. What is real is the parsing logic and the Cypher, because those are
where the first draft was wrong - it sent the cover page to the model and built
relationship types out of role text.
"""

from __future__ import annotations

import pytest

from src import config, relate_db
from src.ingestion import ingest_sec_relationships as sec
from src.ingestion import ingest_stock_data as stock_ing
from src.schema import (
    REL_OWNS,
    REL_SUPPLIES_TO,
    REL_WORKS_FOR,
    STOCK,
    SUBSIDIARY,
    SUPPLIER,
)

SNAPSHOT = {
    "ticker": "NVDA",
    "name": "NVIDIA Corporation",
    "sector": "Technology",
    "industry": "Semiconductors",
    "market_cap": 3.4e12,
    "pe_ratio": 55.2,
    "current_price": 150.0,
    "aliases": ["Nvidia"],
}

TSM = {**SNAPSHOT, "ticker": "TSM", "name": "Taiwan Semiconductor Manufacturing Company Limited", "aliases": ["TSMC"]}


def _seed(conn, *snapshots):
    for snapshot in snapshots:
        stock_ing.upsert_stock(conn, snapshot)


# -- section selection -------------------------------------------------------


def test_window_is_taken_around_the_heading_not_the_document_head():
    cover = "COVER PAGE BOILERPLATE " * 300
    body = " Nominees for Election of Directors: Alice Smith, Jane Doe " + "detail " * 200
    document = cover + body

    window = sec.section_windows(document, sec.PROXY_PATTERNS, size=600)

    assert "Nominees for Election" in window
    assert "Alice Smith" in window
    # Bounded: the window is the requested size plus the small run-in margin,
    # not the whole (cover-page dominated) document.
    assert len(window) <= 600 + 200
    assert len(window) < len(document) / 10


def test_no_heading_means_no_window():
    assert sec.section_windows("just an unrelated paragraph", sec.PROXY_PATTERNS) == ""


def test_windows_are_capped():
    text = " ".join(f"subsidiaries of the registrant {index} " + "x" * 80 for index in range(6))
    window = sec.section_windows(
        text, sec.ANNUAL_SUBSIDIARY_PATTERNS, size=120, max_windows=2
    )
    assert window.count("\n\n") <= 1


def test_window_size_comes_from_config(monkeypatch):
    monkeypatch.setattr(config, "SEC_WINDOW_CHARS", 50)
    text = "board of directors " + "y" * 500
    assert len(sec.section_windows(text, sec.PROXY_PATTERNS)) <= 50 + 200


# -- name cleaning -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ([" Alice ", "alice", "", None, 42, "Bob"], ["Alice", "Bob"]),
        (["x" * 200], []),
        ("Alice", []),
        ([], []),
    ],
)
def test_clean_names(raw, expected):
    assert sec._clean_names(raw, limit=5) == expected


def test_clean_names_enforces_the_limit():
    assert len(sec._clean_names([f"Name {index}" for index in range(20)], limit=3)) == 3


# -- LLM extraction ----------------------------------------------------------


def test_officers_are_normalised_and_junk_is_dropped(monkeypatch):
    monkeypatch.setattr(
        sec.llm,
        "chat_json",
        lambda *a, **k: {
            "executives": [
                {"name": "Jane Doe", "role": "CEO"},
                {"name": "  ", "role": "CFO"},
                {"not": "a person"},
                {"name": "John Smith", "role": ""},
            ]
        },
    )
    officers = sec.extract_officers("window text", "NVDA")
    assert officers == [
        {"name": "Jane Doe", "role": "CEO"},
        {"name": "John Smith", "role": "Officer"},
    ]


def test_empty_window_never_calls_the_model(monkeypatch):
    monkeypatch.setattr(
        sec.llm, "chat_json", lambda *a, **k: pytest.fail("LLM must not be called")
    )
    assert sec.extract_officers("", "NVDA") == []
    assert sec.extract_structure("", "NVDA") == {"subsidiaries": [], "suppliers": []}


def test_structure_survives_a_malformed_model_response(monkeypatch):
    def bad_json(*args, **kwargs):
        raise ValueError("not json")

    monkeypatch.setattr(sec.llm, "chat_json", bad_json)
    assert sec.extract_structure("window", "NVDA") == {
        "subsidiaries": [],
        "suppliers": [],
    }


# -- supplier resolution -----------------------------------------------------


def test_resolver_matches_ticker_and_alias(conn):
    _seed(conn, SNAPSHOT, TSM)
    assert sec.resolve_tracked_stock(conn, "TSMC") == "TSM"
    assert sec.resolve_tracked_stock(conn, "Nvidia") == "NVDA"
    assert sec.resolve_tracked_stock(conn, "nvidia corporation") == "NVDA"


def test_resolver_rejects_generic_phrases(conn):
    _seed(conn, SNAPSHOT, TSM)
    assert sec.resolve_tracked_stock(conn, "Cloud Infrastructure Providers") is None
    assert sec.resolve_tracked_stock(conn, "AI") is None


# -- writes ------------------------------------------------------------------


def test_write_subsidiary_is_idempotent(conn):
    _seed(conn, SNAPSHOT)
    for _ in range(2):
        sec.write_subsidiary(conn, "NVDA", "NVIDIA Global Holdings")
    assert relate_db.counts(conn)[SUBSIDIARY] == 1
    edges = relate_db.run(conn, f"MATCH ()-[r:{REL_OWNS}]->() RETURN count(r) AS n")
    assert edges[0]["n"] == 1


def test_write_supplier_links_tracked_stock_instead_of_duplicating_it(conn):
    _seed(conn, SNAPSHOT, TSM)
    description = sec.write_supplier(conn, "NVDA", "TSMC")

    assert "tracked stock" in description
    assert relate_db.counts(conn).get(SUPPLIER, 0) == 0
    edge = relate_db.run(
        conn,
        f"""
        MATCH (sup:{STOCK} {{ticker: 'TSM'}})-[r:{REL_SUPPLIES_TO}]->(s:{STOCK} {{ticker: 'NVDA'}})
        RETURN r.source AS source, r.critical AS critical
        """,
    )
    assert edge and edge[0]["source"] == "10-K" and edge[0]["critical"] is True


def test_write_supplier_creates_a_vendor_node_when_untracked(conn):
    _seed(conn, SNAPSHOT)
    description = sec.write_supplier(conn, "NVDA", "Some Private Foundry")

    assert "unlisted supplier" in description
    assert relate_db.counts(conn)[SUPPLIER] == 1
    # Vendors share SUPPLIES_TO with tracked stocks, so one pattern finds both.
    edges = relate_db.run(
        conn,
        f"MATCH (v:{SUPPLIER})-[r:{REL_SUPPLIES_TO}]->(:{STOCK}) "
        "RETURN r.source AS source",
    )
    assert edges and edges[0]["source"] == "10-K"


def test_write_supplier_does_not_duplicate_an_existing_curated_edge(conn):
    _seed(conn, SNAPSHOT, TSM)
    stock_ing.write_relationship(conn, "TSM", "NVDA", REL_SUPPLIES_TO)

    sec.write_supplier(conn, "NVDA", "TSMC")

    edges = relate_db.run(
        conn, f"MATCH ()-[r:{REL_SUPPLIES_TO}]->() RETURN count(r) AS n"
    )
    assert edges[0]["n"] == 1


def test_write_officer_preserves_a_curated_role(conn):
    _seed(conn, SNAPSHOT)
    stock_ing.write_person(conn, "Jensen Huang", "CEO", "NVDA")

    sec.write_officer(conn, "NVDA", "Jensen Huang", "Board Member")

    role = relate_db.run(
        conn, "MATCH (p:Person {name: 'Jensen Huang'}) RETURN p.role AS role"
    )[0]["role"]
    assert role == "CEO"
    source = relate_db.run(
        conn, f"MATCH ()-[r:{REL_WORKS_FOR}]->() RETURN r.source AS source"
    )[0]["source"]
    assert source == "DEF 14A"


def test_write_officer_fills_an_empty_role(conn):
    _seed(conn, SNAPSHOT)
    sec.write_officer(conn, "NVDA", "Jane Doe", "Director")
    role = relate_db.run(
        conn, "MATCH (p:Person {name: 'Jane Doe'}) RETURN p.role AS role"
    )[0]["role"]
    assert role == "Director"


# -- person identity ---------------------------------------------------------


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("Jensen Huang", "Jen-Hsun Huang", True),   # the real EDGAR spelling
        ("Jane Doe", "jane doe", True),
        ("Mark A. Stevens", "Mark Stevens", True),
        ("John Smith", "James Smith", False),        # same surname, different person
        ("Jensen Huang", "Jensen Huang", True),
        ("Jensen", "Jensen Huang", False),           # too little information
    ],
)
def test_same_person_heuristic(left, right, expected):
    assert sec.same_person(left, right) is expected


def test_officer_reuses_the_curated_person_node(conn):
    """Regression: EDGAR's 'Jen-Hsun Huang' duplicated the curated CEO node."""
    _seed(conn, SNAPSHOT)
    stock_ing.write_person(conn, "Jensen Huang", "CEO", "NVDA")

    sec.write_officer(conn, "NVDA", "Jen-Hsun Huang", "CEO")

    people = relate_db.run(conn, "MATCH (p:Person) RETURN p.name AS name")
    assert [row["name"] for row in people] == ["Jensen Huang"]
    edges = relate_db.run(conn, f"MATCH ()-[r:{REL_WORKS_FOR}]->() RETURN count(r) AS n")
    assert edges[0]["n"] == 1


def test_person_resolution_is_scoped_to_the_company(conn):
    """A same-named person at another company must not be reused."""
    _seed(conn, SNAPSHOT, TSM)
    stock_ing.write_person(conn, "Jensen Huang", "CEO", "TSM")

    sec.write_officer(conn, "NVDA", "Jensen Huang", "CEO")

    edges = relate_db.run(
        conn,
        f"MATCH (p:Person {{name: 'Jensen Huang'}})-[:{REL_WORKS_FOR}]->(s:{STOCK}) "
        "RETURN s.ticker AS ticker ORDER BY ticker",
    )
    assert [row["ticker"] for row in edges] == ["NVDA", "TSM"]


def test_reset_removes_only_sec_nodes(conn):
    _seed(conn, SNAPSHOT)
    sec.write_subsidiary(conn, "NVDA", "Some Sub")
    sec.write_supplier(conn, "NVDA", "Some Vendor")

    sec.reset_sec_nodes(conn)

    counts = relate_db.counts(conn)
    assert counts.get(SUBSIDIARY, 0) == 0
    assert counts.get(SUPPLIER, 0) == 0
    assert counts[STOCK] == 1  # stocks and their edges survive


# -- cache layout ------------------------------------------------------------


def test_filing_path_picks_the_newest_document(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SEC_CACHE_DIR", tmp_path)
    form_dir = tmp_path / "sec-edgar-filings" / "NVDA" / "10-K"
    old = form_dir / "0001045810-24-000001"
    new = form_dir / "0001045810-26-000084"
    for directory in (old, new):
        directory.mkdir(parents=True)
    (old / "primary-document.html").write_text("old", encoding="utf-8")
    (new / "primary-document.html").write_text("new", encoding="utf-8")
    # mtime, not the accession string, decides which is newest.
    import os

    os.utime(old / "primary-document.html", (1_000_000, 1_000_000))

    assert sec.filing_path("NVDA", "10-K").read_text(encoding="utf-8") == "new"


def test_filing_path_returns_none_when_nothing_is_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SEC_CACHE_DIR", tmp_path)
    assert sec.filing_path("NVDA", "10-K") is None


def test_html_to_text_flattens_markup():
    html = "<html><body><p>Board of <b>Directors</b></p><script>x()</script></body></html>"
    assert sec.html_to_text(html) == "Board of Directors"
