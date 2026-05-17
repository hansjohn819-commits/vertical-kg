"""Unit tests for the composer history-budget guard (§16.1).

These verify the three states of `_maybe_summarize_history`:

  1. history is None / empty → returned as-is, no LLM call
  2. (history + question) under budget → returned as-is, no LLM call
  3. (history + question) over budget → LLM call, replaced with one
     synthetic system-message containing the summary

A fourth path (LLM call raises / returns empty) degrades to dropping
history rather than crashing; that's also covered here so we don't
silently regress to "QA path dies on long conversations."
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.modules.m2_qa import (
    HISTORY_QUESTION_BUDGET_TOKENS,
    _maybe_summarize_history,
)


# --- helpers -----------------------------------------------------------

def _fake_resp(text: str):
    """Shape the LocalClient.chat() return enough for the helper to read
    .choices[0].message.content. SimpleNamespace is enough — nothing
    pydantic-fancy is touched on that path."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _mk_client(summary_text: str | None = "SUMMARY", raises: bool = False):
    client = MagicMock()
    if raises:
        client.chat.side_effect = RuntimeError("simulated backend wedge")
    else:
        client.chat.return_value = _fake_resp(summary_text or "")
    return client


# --- tests -------------------------------------------------------------

def test_none_history_returns_none_and_no_llm_call():
    client = _mk_client()
    out = _maybe_summarize_history(client, None, "anything")
    assert out is None
    client.chat.assert_not_called()


def test_empty_history_returns_empty_and_no_llm_call():
    client = _mk_client()
    out = _maybe_summarize_history(client, [], "anything")
    assert out == []
    client.chat.assert_not_called()


def test_short_history_passes_through_unchanged():
    client = _mk_client()
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello, how can I help?"},
    ]
    out = _maybe_summarize_history(client, history, "What is X?")
    assert out is history  # same list object, untouched
    client.chat.assert_not_called()


def test_over_budget_triggers_summarization():
    # Pad to exceed the budget. count_tokens uses ~4 chars/token so a
    # single message at 4 * BUDGET chars is comfortably over.
    bloat = "x " * (HISTORY_QUESTION_BUDGET_TOKENS * 3)
    history = [{"role": "user", "content": bloat}]
    client = _mk_client(summary_text="condensed transcript")
    out = _maybe_summarize_history(client, history, "follow up question")

    assert client.chat.call_count == 1
    # The summary call is a 2-message exchange (system + user transcript).
    msgs = client.chat.call_args.kwargs["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "compressor" in msgs[0]["content"].lower()
    assert msgs[1]["role"] == "user"
    # Transcript carries the original role label so summarizer sees who
    # said what.
    assert msgs[1]["content"].startswith("USER:")

    # Output is a single synthetic system message with the summary body.
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["role"] == "system"
    assert "condensed transcript" in out[0]["content"]


def test_thinking_off_for_summary_call():
    bloat = "x " * (HISTORY_QUESTION_BUDGET_TOKENS * 3)
    history = [{"role": "user", "content": bloat}]
    client = _mk_client(summary_text="ok")
    _maybe_summarize_history(client, history, "q")
    assert client.chat.call_args.kwargs.get("thinking") is False


def test_summary_failure_drops_history():
    # If the LLM wedges, the helper should degrade to dropping history
    # rather than propagating the exception. Losing transcript context
    # is a downgrade; crashing the QA call is a failure.
    bloat = "x " * (HISTORY_QUESTION_BUDGET_TOKENS * 3)
    history = [{"role": "user", "content": bloat}]
    client = _mk_client(raises=True)
    out = _maybe_summarize_history(client, history, "q")
    assert out is None


def test_empty_summary_string_also_drops():
    # LLM returned an empty string — treat the same as failure.
    bloat = "x " * (HISTORY_QUESTION_BUDGET_TOKENS * 3)
    history = [{"role": "user", "content": bloat}]
    client = _mk_client(summary_text="")
    out = _maybe_summarize_history(client, history, "q")
    assert out is None


def test_custom_budget_can_force_summarization():
    # A short conversation should normally pass through, but if the
    # caller specifies a tiny budget the summarizer must still fire.
    history = [
        {"role": "user", "content": "this is a moderately long message"},
        {"role": "assistant", "content": "and a moderately long reply"},
    ]
    client = _mk_client(summary_text="tiny")
    out = _maybe_summarize_history(client, history, "q", budget=5)
    assert client.chat.call_count == 1
    assert out and out[0]["role"] == "system"
