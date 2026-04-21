"""LLM routing.

Current build: single-backend (local OpenAI-compatible). Cross-provider
fallback for AGENT_LLM deferred until local capability proves insufficient
(guide §6.3, 2026-04-21).
"""

import os

from dotenv import load_dotenv

from .local_client import LocalClient

load_dotenv()

_CACHE: dict[tuple[str, str], LocalClient] = {}


def _role_backend(role: str) -> str:
    """AGENT_LLM falls back to BACKEND_LLM when unset."""
    if role == "agent":
        return os.getenv("AGENT_LLM") or os.getenv("BACKEND_LLM", "local")
    return os.getenv("BACKEND_LLM", "local")


def _build(name: str) -> LocalClient:
    if name == "local":
        return LocalClient()
    raise ValueError(
        f"Unknown LLM backend {name!r}. Only 'local' is built in the current "
        f"scope (see guide §6.3)."
    )


def get_client(role: str = "backend") -> LocalClient:
    """role ∈ {'backend', 'agent'}."""
    name = _role_backend(role)
    key = (role, name)
    if key not in _CACHE:
        _CACHE[key] = _build(name)
    return _CACHE[key]


def validate_env() -> dict:
    """Call on startup. Raises on hard config error; returns a status dict."""
    backend = _role_backend("backend")
    agent = _role_backend("agent")
    for role, name in (("BACKEND_LLM", backend), ("AGENT_LLM", agent)):
        if name != "local":
            raise RuntimeError(
                f"{role}={name!r} not supported. Only 'local' is currently built."
            )
    client = get_client("backend")
    return {
        "backend": backend,
        "agent": agent,
        "base_url": client.base_url,
        "model": client.model,
        "reachable": client.ping(),
    }
