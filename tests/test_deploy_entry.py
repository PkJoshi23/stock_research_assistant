"""The Cloud entry point must actually render the real app.

`streamlit_app.py` is a `runpy` shim around `src/ui/app.py`. If that shim breaks,
Streamlit Cloud boots to a blank page while every other test still passes - so it
gets its own test, pointed at a temporary store so it never touches the demo db.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from src import config


def test_cloud_entry_point_renders(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "graph.rl")

    app = AppTest.from_file("streamlit_app.py", default_timeout=180)
    app.run()

    assert not app.exception, [str(exc.value) for exc in app.exception]
    # The app's sidebar is the reliable structural signal that app.py ran.
    assert "Graph" in [header.value for header in app.subheader]


def test_cloud_entry_point_reports_an_empty_graph(tmp_path, monkeypatch, capsys):
    """A fresh deploy with no committed db should say so, not crash."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "graph.rl")

    app = AppTest.from_file("streamlit_app.py", default_timeout=180)
    app.run()

    assert not app.exception
    warnings = [w.value for w in app.warning]
    assert any("empty" in text.lower() for text in warnings)
