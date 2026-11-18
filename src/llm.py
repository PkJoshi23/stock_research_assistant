"""The only place that talks to DeepSeek.

Before this module there were four independent ``OpenAI(...)`` clients (router,
generator, executor, sidebar), each hardcoding the base URL and each falling
back to the literal string ``"your_deepseek_api_key_here"``. That combination
has two bad failure modes:

  * a missing key produces a 401 from step 3 instead of a clear error up front;
  * the key typed into the Streamlit sidebar was used only for the final
    summary, so steps 1-3 silently kept using the env key - a judge typing
    their own key would see nothing change.

One client, resolved lazily at call time, fixes both: ``set_api_key`` lets the
UI override, and ``require_secret`` fails loudly when nothing is configured.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI

from . import config

#: Set by the Streamlit sidebar (or any caller) to override the env key.
_api_key_override: str | None = None

_client: OpenAI | None = None

#: Per-thread usage accumulator. A module global would mix concurrent Streamlit
#: sessions into one number; thread-local keeps each question's cost its own.
_local = threading.local()

_EMPTY_USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "ms": 0.0}


def start_trace() -> None:
    """Begin recording token usage and latency for the current thread."""
    _local.usage = dict(_EMPTY_USAGE)


def finish_trace() -> dict[str, Any]:
    """Stop recording and return the totals (always a complete dict)."""
    usage = getattr(_local, "usage", None) or dict(_EMPTY_USAGE)
    _local.usage = None
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    usage["ms"] = round(usage["ms"], 1)
    return usage


def _record(prompt_tokens: int, completion_tokens: int, elapsed_ms: float) -> None:
    usage = getattr(_local, "usage", None)
    if usage is None:
        return  # no active trace: telemetry is opt-in, not always-on
    usage["calls"] += 1
    usage["prompt_tokens"] += int(prompt_tokens or 0)
    usage["completion_tokens"] += int(completion_tokens or 0)
    usage["ms"] += elapsed_ms


def set_api_key(api_key: str | None) -> None:
    """Override the API key for the rest of the process.

    Passing ``None`` clears the override and falls back to the environment.
    A changed key invalidates the cached client.
    """
    global _api_key_override, _client
    _api_key_override = api_key or None
    _client = None


def get_client() -> OpenAI:
    """Return a cached client, creating it on first use.

    Lazy on purpose: importing this module must not require a key, otherwise
    every CLI tool and test would need one just to import the package.
    """
    global _client
    if _client is None:
        key = _api_key_override or config.require_secret("DEEPSEEK_API_KEY")
        _client = OpenAI(api_key=key, base_url=config.DEEPSEEK_BASE_URL)
    return _client


def chat(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> str:
    """Single-turn completion returning the raw text content."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": config.DEEPSEEK_CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        # DeepSeek honours the OpenAI ``response_format`` switch, which is the
        # only reliable way to stop it wrapping JSON in prose.
        kwargs["response_format"] = {"type": "json_object"}

    started = time.perf_counter()
    # One retry for transport failures only. A 401/429 is a different problem and
    # retrying it would just spend the user's quota twice; a DNS or edge blip
    # (observed as APIConnectionError on this network) usually clears instantly.
    for attempt in (1, 2):
        try:
            response = get_client().chat.completions.create(**kwargs)
            break
        except (APIConnectionError, APITimeoutError):
            if attempt == 2:
                raise
            time.sleep(2)
    elapsed_ms = (time.perf_counter() - started) * 1000

    usage = getattr(response, "usage", None)
    _record(
        getattr(usage, "prompt_tokens", 0),
        getattr(usage, "completion_tokens", 0),
        elapsed_ms,
    )
    return (response.choices[0].message.content or "").strip()


def chat_json(prompt: str, *, system: str | None = None) -> dict[str, Any]:
    """Completion parsed as a JSON object.

    Raises ``ValueError`` when the model returns non-JSON despite ``json_mode``;
    callers decide whether that is fatal or a cue to use heuristics.
    """
    raw = chat(prompt, system=system, temperature=0.0, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model did not return JSON: {raw[:200]!r}") from exc


_CYPHER_FENCE = re.compile(r"^```(?:cypher|sql)?\s*|\s*```$", re.IGNORECASE)


def extract_cypher(text: str) -> str:
    """Strip markdown fences and stray prose from a generated query.

    Models obey "raw Cypher only" most of the time; the rest of the time they
    wrap it in fences or prefix it with "Here is the query:". Stripping here
    means both the generator and the auto-correction loop share one cleaner.
    """
    cleaned = _CYPHER_FENCE.sub("", text.strip()).strip()

    # Keep the first statement only: a trailing "; MATCH ..." is never intended
    # and would execute a second query the guard never saw.
    if ";" in cleaned:
        cleaned = cleaned.split(";", 1)[0].strip()

    # Drop a leading sentence if the model ignored the format rule.
    match = re.search(r"\b(MATCH|CALL|WITH|UNWIND|RETURN|OPTIONAL\s+MATCH)\b", cleaned)
    if match and match.start() > 0:
        cleaned = cleaned[match.start():]

    return cleaned
