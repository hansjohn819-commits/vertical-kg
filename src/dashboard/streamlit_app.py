"""Streamlit chat front-end.

Single-page UI that wraps `GraphAgent` for Q&A and graph management.
The agent's tool box is the single control surface — the user asks
("ingest q1.txt", "what does the graph know about X?", "run a sleep
pass"), and the agent picks the right tool. There is no separate
"ingest mode": knowledge enters the graph only via the `ingest_file`
tool against documents the user has placed under `data/raw/`.

Chat history is fed back to the agent through a rolling token window
(≤ HISTORY_BUDGET_TOKENS) so prompt size stays bounded regardless of
how long the session runs (§12.5.2).

Run:
    streamlit run src/dashboard/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

import streamlit as st

from src.graph.instance import GraphInstance
from src.graph.tokens import count_tokens
from src.modules.m2_qa_agent import GraphAgent

WORKSPACE = Path(__file__).resolve().parents[2]
INSTANCE_DIR = WORKSPACE / "data" / "production"
ONTOLOGY_PATH = WORKSPACE / "ontology.md"

HISTORY_BUDGET_TOKENS = 3_000  # guide §12.5.2 chat-history allotment


@st.cache_resource(show_spinner=False)
def _load_instance() -> GraphInstance:
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    return GraphInstance(name="production", storage_path=INSTANCE_DIR, ontology_path=ONTOLOGY_PATH)


def _windowed_history(messages: list[dict]) -> list[dict]:
    """Keep the newest messages that fit HISTORY_BUDGET_TOKENS."""
    kept: list[dict] = []
    tok = 0
    for m in reversed(messages):
        t = count_tokens(m.get("content") or "")
        if tok + t > HISTORY_BUDGET_TOKENS:
            break
        kept.append(m)
        tok += t
    return list(reversed(kept))


def main() -> None:
    st.set_page_config(page_title="Self-Evolving KG", layout="wide")
    gi = _load_instance()

    with st.sidebar:
        if st.button("Clear chat"):
            st.session_state.pop("messages", None)
            st.rerun()

    st.title("Self-Evolving Knowledge Graph")
    st.caption(
        "Ask the agent about the graph, or tell it to ingest a file you've "
        "placed under data/raw/."
    )

    if gi.sleep_pass_running:
        st.warning("Sleep pass is running; chat is paused until it finishes.")

    st.session_state.setdefault("messages", [])

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask the graph, or ask to ingest a file…")
    if not prompt:
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    agent = GraphAgent(gi)
    history = _windowed_history(st.session_state.messages[:-1])
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply = agent.call(prompt, history=history)
            except Exception as exc:
                reply = f"Error: {exc}"
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    gi.save()
    st.rerun()


if __name__ == "__main__":
    main()
