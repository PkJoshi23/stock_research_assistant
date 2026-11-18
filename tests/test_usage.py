"""Token and latency telemetry.

Cost per answer is a credibility number in a demo ("each question costs a
fraction of a cent"), so it has to be measured rather than estimated. The
accumulator is thread-local on purpose: two Streamlit sessions must not sum
their tokens into one figure.
"""

from __future__ import annotations

import types

from src import llm


class _FakeCompletions:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def create(self, **kwargs):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))],
            usage=types.SimpleNamespace(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
            ),
        )


def _fake_client(prompt_tokens: int = 11, completion_tokens: int = 7):
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_FakeCompletions(prompt_tokens, completion_tokens))
    )


def test_trace_records_tokens_and_time(monkeypatch):
    monkeypatch.setattr(llm, "_client", _fake_client(11, 7))

    llm.start_trace()
    assert llm.chat("hello") == "ok"
    usage = llm.finish_trace()

    assert usage["calls"] == 1
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 7
    assert usage["total_tokens"] == 18
    assert usage["ms"] >= 0


def test_usage_accumulates_across_calls(monkeypatch):
    monkeypatch.setattr(llm, "_client", _fake_client(10, 5))

    llm.start_trace()
    llm.chat("one")
    llm.chat("two")
    usage = llm.finish_trace()

    assert usage["calls"] == 2
    assert usage["total_tokens"] == 30


def test_calls_without_a_trace_are_not_recorded(monkeypatch):
    monkeypatch.setattr(llm, "_client", _fake_client(5, 5))
    llm.finish_trace()  # clear any state from another test

    llm.chat("outside a trace")
    assert llm.finish_trace()["calls"] == 0


def test_finish_trace_resets_the_accumulator(monkeypatch):
    monkeypatch.setattr(llm, "_client", _fake_client())

    llm.start_trace()
    llm.chat("one")
    llm.finish_trace()

    assert llm.finish_trace()["calls"] == 0


def test_missing_usage_field_does_not_crash(monkeypatch):
    class NoUsage:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="x"))]
            )

    monkeypatch.setattr(llm, "_client", types.SimpleNamespace(chat=types.SimpleNamespace(completions=NoUsage())))

    llm.start_trace()
    llm.chat("hi")
    usage = llm.finish_trace()

    assert usage["calls"] == 1
    assert usage["total_tokens"] == 0
