"""The unknown-ticker path in the deployed UI.

A coverage miss must render the helpful message *and* offer the SEC fetch; clicking
it must ingest and re-answer. `fast_sec.ingest` is replaced, so nothing here touches
the network.
"""

from __future__ import annotations

import pytest

from src import config, coverage, relate_db
from src.ingestion import fast_sec

REGISTRY = [
    {"ticker": "AAOI", "title": "APPLIED OPTOELECTRONICS, INC.", "cik": "0001158114"},
]


@pytest.fixture
def ui(tmp_path, monkeypatch, seed_graph):
    """Seed the store the app will open, and register AAOI as a known ticker."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "graph.rl")
    monkeypatch.setattr(
        coverage.ticker_directory, "load_directory", lambda **kwargs: REGISTRY
    )

    seeding = relate_db.GraphSession()
    seed_graph(seeding.run)
    seeding.close()

    app = AppTest.from_file("streamlit_app.py", default_timeout=240)
    app.run()
    return app


def test_unknown_ticker_shows_the_refusal_and_a_fetch_button(ui):
    ui.chat_input[0].set_value("who supplies to aaoi?").run()

    assert not ui.exception
    rendered = " ".join(markdown.value for markdown in ui.markdown)
    assert "AAOI" in rendered
    assert "don't track" in rendered.lower()

    keys = [button.key for button in ui.button]
    assert "fetch-sec-AAOI" in keys


def test_fetch_button_ingests_and_answers_again(ui, monkeypatch):
    ui.chat_input[0].set_value("who supplies to aaoi?").run()

    calls: dict = {}

    def fake_ingest(conn, ticker, **kwargs):
        calls["ticker"] = ticker
        calls["write"] = kwargs.get("write")
        return fast_sec.FastReport(
            ticker=ticker,
            form="10-K",
            filed="2026-02-26",
            suppliers=["Acme Optics"],
            elapsed_ms=120,
        )

    monkeypatch.setattr(fast_sec, "ingest", fake_ingest)

    button = next(item for item in ui.button if item.key == "fetch-sec-AAOI")
    button.click()
    ui.run()

    assert calls == {"ticker": "AAOI", "write": True}
    assert any("Added AAOI" in message.value for message in ui.success)
