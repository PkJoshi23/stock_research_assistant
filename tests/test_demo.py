"""The readiness check: does it tell the truth about an empty graph?

An empty store is the default state of a fresh clone, and the failure mode is
subtle - the app starts fine and answers "no data matching that" to everything.
"""

from __future__ import annotations

import json

import pytest

from src import demo, relate_db


def test_empty_store_reports_not_ready(db_path, capsys):
    code = demo.main([])
    out = capsys.readouterr().out

    assert code == 1
    assert "EMPTY" in out
    assert "python -m src.demo --seed" in out


def test_populated_store_reports_ready(db_path, seed_graph, capsys):
    seeding = relate_db.GraphSession()
    seed_graph(seeding.run)
    seeding.close()

    code = demo.main([])
    out = capsys.readouterr().out

    assert code == 0
    assert "[ ok ]" in out
    assert "streamlit run src/ui/app.py" in out
    assert "Who supplies Nvidia?" in out


def test_json_output_is_machine_readable(db_path, seed_graph, capsys):
    seeding = relate_db.GraphSession()
    seed_graph(seeding.run)
    seeding.close()

    demo.main(["--json"])
    report = json.loads(capsys.readouterr().out)

    assert report["nodes"] > 0
    assert report["counts"]["Stock"] == 4
    assert "stock_ticker_idx" in report["indexes"]


def test_missing_api_key_is_warned_but_not_fatal(db_path, seed_graph, monkeypatch, capsys):
    """Graph questions work offline; the check must not block a keyless demo."""
    monkeypatch.setattr("src.config.DEEPSEEK_API_KEY", None)
    seeding = relate_db.GraphSession()
    seed_graph(seeding.run)
    seeding.close()

    code = demo.main([])
    out = capsys.readouterr().out

    assert code == 0
    assert "DEEPSEEK_API_KEY" in out
    assert "MISSING" in out


def test_discover_flag_runs_the_crawler_with_the_given_bounds(
    db_path, seed_graph, monkeypatch
):
    """One command must be able to build the graph *and* grow the supply chain."""
    seeding = relate_db.GraphSession()
    seed_graph(seeding.run)
    seeding.close()

    from src.ingestion import discover_supply_chain

    calls: dict = {}
    monkeypatch.setattr(
        discover_supply_chain, "main", lambda argv=None: calls.setdefault("argv", argv) or 0
    )

    demo.main(["--discover", "--depth", "2", "--max-nodes", "7"])

    assert calls["argv"] == ["--depth", "2", "--max-nodes", "7"]


def test_discover_flag_is_off_by_default(db_path, seed_graph, monkeypatch):
    seeding = relate_db.GraphSession()
    seed_graph(seeding.run)
    seeding.close()

    from src.ingestion import discover_supply_chain

    monkeypatch.setattr(
        discover_supply_chain, "main", lambda argv=None: pytest.fail("must not crawl")
    )

    assert demo.main([]) == 0
