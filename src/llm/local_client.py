"""Thin wrapper over an OpenAI-compatible local endpoint.

Any backend that speaks the OpenAI /v1/chat/completions schema works —
swapping the underlying server only changes env vars, not code.
"""

import os

from openai import OpenAI


class LocalClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        context_tokens: int | None = None,
    ):
        self.base_url = base_url or os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
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
    ):
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        return self._client.chat.completions.create(**kwargs)

    def ping(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            return False
