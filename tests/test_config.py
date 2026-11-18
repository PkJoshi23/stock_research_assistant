"""Configuration sourcing: environment first, then Streamlit Cloud secrets.

Cloud has no `.env`, so the app must read `st.secrets`. These tests prove the
precedence and that a missing secrets file never breaks a CLI import.
"""

from __future__ import annotations

import sys
import types

from src import config


def _fake_streamlit(secrets: dict[str, str]) -> types.SimpleNamespace:
    return types.SimpleNamespace(secrets=secrets)


def test_environment_wins_over_streamlit_secrets(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit({"DEEPSEEK_API_KEY": "from-cloud"}))

    assert config._setting("DEEPSEEK_API_KEY") == "from-env"


def test_streamlit_secrets_are_used_when_env_is_absent(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit({"DEEPSEEK_API_KEY": "from-cloud"}))

    assert config._setting("DEEPSEEK_API_KEY") == "from-cloud"


def test_default_is_returned_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv("SOME_UNSET_SETTING", raising=False)
    monkeypatch.delitem(sys.modules, "streamlit", raising=False)

    assert config._setting("SOME_UNSET_SETTING", "fallback") == "fallback"


def test_missing_secrets_file_is_not_fatal(monkeypatch):
    """Accessing st.secrets with no secrets.toml raises; config must absorb it."""

    class ExplodingSecrets:
        def __contains__(self, name):
            raise RuntimeError("no secrets file")

    monkeypatch.delenv("SOME_UNSET_SETTING", raising=False)
    monkeypatch.setitem(sys.modules, "streamlit", types.SimpleNamespace(secrets=ExplodingSecrets()))

    assert config._setting("SOME_UNSET_SETTING", "fallback") == "fallback"
