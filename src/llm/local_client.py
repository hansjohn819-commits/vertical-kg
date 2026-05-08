"""Thin wrapper over an OpenAI-compatible local endpoint.

Any backend that speaks the OpenAI /v1/chat/completions schema works —
swapping the underlying server only changes env vars, not code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterator

from openai import OpenAI


@dataclass
class ToolCallDelta:
    """Accumulated tool call from a streamed response."""

    id: str
    name: str
    arguments: str


@dataclass
class StreamEvent:
    """A single event yielded by ``chat_stream``.

    Exactly one of *token* or *tool_calls* is set per event.
    """

    token: str | None = None
    tool_calls: list[ToolCallDelta] = field(default_factory=list)


class LocalClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        context_tokens: int | None = None,
    ):
        self.base_url = base_url or os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8080/v1")
        self.api_key = api_key or os.getenv("LOCAL_LLM_API_KEY", "not-needed")
        self.model = model or os.getenv("LOCAL_LLM_MODEL", "google/gemma-4-26b-a4b")
        # Input + output share this budget. See guide §12.5.
        self.context_tokens = context_tokens or int(os.getenv("LOCAL_LLM_CONTEXT_TOKENS", "40000"))
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.7,
        timeout: float = 120,
        thinking: bool = True,
    ):
        """`thinking=False` disables Gemma 4 native thinking via llama.cpp's
        `chat_template_kwargs` passthrough. Probe-tested 2026-05-06 (§15
        changelog): 49.67s → 3.18s, ~15.6× speedup, no side effects on
        non-thinking models (servers drop unrecognized chat_template_kwargs).
        Use False for: (a) external fast_query path (Phase C), (b) M4d
        link_form judge — thinking-on rationalizes trivial relations into
        plausible-sounding edges, hurting precision. Default True elsewhere
        (M1 extraction / M2 agent / M4b judges benefit from reasoning).
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if not thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        return self._client.chat.completions.create(**kwargs)

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.7,
        timeout: float = 120,
        thinking: bool = True,
    ) -> Iterator[StreamEvent]:
        """Streaming variant of :meth:`chat`.

        Yields :class:`StreamEvent` objects.  Content tokens arrive as
        ``StreamEvent(token=...)``; when the model emits tool calls the
        chunks are accumulated internally and a single
        ``StreamEvent(tool_calls=[...])`` is yielded at ``finish_reason ==
        "tool_calls"``.
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if not thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        pending_tool_calls: dict[int, ToolCallDelta] = {}

        for chunk in self._client.chat.completions.create(**kwargs):
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue
            delta = choice.delta

            if delta.content:
                yield StreamEvent(token=delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in pending_tool_calls:
                        pending_tool_calls[idx] = ToolCallDelta(
                            id=tc_delta.id or "",
                            name=tc_delta.function.name or "" if tc_delta.function else "",
                            arguments="",
                        )
                    tc = pending_tool_calls[idx]
                    if tc_delta.id and not tc.id:
                        tc.id = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc.name = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc.arguments += tc_delta.function.arguments

            if choice.finish_reason == "tool_calls" and pending_tool_calls:
                yield StreamEvent(
                    tool_calls=[pending_tool_calls[i] for i in sorted(pending_tool_calls)],
                )
                pending_tool_calls.clear()

        if pending_tool_calls:
            yield StreamEvent(
                tool_calls=[pending_tool_calls[i] for i in sorted(pending_tool_calls)],
            )

    def ping(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            return False
