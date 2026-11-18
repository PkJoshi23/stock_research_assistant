"""The deployed UI must never leak the server-side API key into the page.

Streamlit serialises widget values into the DOM, so prefilling the "API key"
field with `config.DEEPSEEK_API_KEY` would hand the credential to every visitor
of a public app - visible only as dots on screen, readable in devtools. This is
the regression test for that.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from src import config

SECRET = "sk-this-is-the-server-key"


def _app(tmp_path, monkeypatch, key: str | None) -> AppTest:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "graph.rl")
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", key)
    app = AppTest.from_file("streamlit_app.py", default_timeout=180)
    app.run()
    return app


def test_server_key_is_never_rendered_into_the_widget(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch, SECRET)

    assert not app.exception
    key_fields = [field for field in app.text_input if "API key" in field.label]
    assert key_fields, "the optional key override should still exist"
    assert all(field.value == "" for field in key_fields)
    # And the secret must not appear anywhere in the rendered text.
    rendered = " ".join(
        str(element.value) for element in list(app.markdown) + list(app.caption) + list(app.text)
    )
    assert SECRET not in rendered


def test_status_caption_says_a_server_key_is_configured(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch, SECRET)

    captions = [caption.value for caption in app.caption]
    assert any("Server key configured" in text for text in captions)


def test_no_key_configured_still_renders(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch, None)

    assert not app.exception
    assert any("API key" in field.label for field in app.text_input)
