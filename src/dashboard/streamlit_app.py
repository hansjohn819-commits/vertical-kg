"""Streamlit chat tab (Phase 7 per guide §10 / §12.5.4).

Single-page UI that wraps `GraphAgent` for Q&A / management and
`GraphInstance.integrate` for pasted-sample ingestion. All other
operations (stats, sleep pass, provenance lookup, etc.) are exposed
implicitly through the agent's tool box — the user asks, the agent
picks the right tool.

Chat history is sent to the agent through a rolling token window
(≤ HISTORY_BUDGET_TOKENS) so prompt size stays bounded regardless of
how long the session runs (§12.5.2).

Run:
    streamlit run src/dashboard/streamlit_app.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

import streamlit as st

from src.graph.instance import GraphInstance
from src.graph.tokens import INTEGRATE_INPUT_SOFT_CAP_TOKENS, count_tokens
from src.modules.m2_qa_agent import GraphAgent
from src.modules.m3_integrate import UserContext

WORKSPACE = Path(__file__).resolve().parents[2]
INSTANCES = {
    "experiment": WORKSPACE / "data" / "experiment",
    "production": WORKSPACE / "data" / "production",
}
ONTOLOGY_PATH = WORKSPACE / "ontology.md"

HISTORY_BUDGET_TOKENS = 3_000  # guide §12.5.2 chat-history allotment


@st.cache_resource(show_spinner=False)
def _load_instance(name: str) -> GraphInstance:
    storage_dir = INSTANCES[name]
    storage_dir.mkdir(parents=True, exist_ok=True)
    return GraphInstance(name=name, storage_path=storage_dir, ontology_path=ONTOLOGY_PATH)


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


def _session_user_ctx() -> UserContext:
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = uuid.uuid4().hex[:8]
    st.session_state.setdefault("turn_counter", 0)
    st.session_state.turn_counter += 1
    return UserContext(
        user_id="dashboard-user",
        conversation_id=st.session_state.conversation_id,
        turn_id=f"t{st.session_state.turn_counter}",
    )


def main() -> None:
    st.set_page_config(page_title="Self-Evolving KG", layout="wide")

    with st.sidebar:
        st.header("Instance")
        name = st.selectbox("Graph", list(INSTANCES.keys()), index=0)
        gi = _load_instance(name)
        stats = gi.stats()
        c1, c2 = st.columns(2)
        c1.metric("Nodes", stats["node_count"])
        c2.metric("Edges", stats["edge_count"])
        with st.expander("Node types", expanded=False):
            st.json(stats.get("node_types", {}))
        with st.expander("Edge types", expanded=False):
            st.json(stats.get("edge_types", {}))

        st.divider()
        mode = st.radio("Mode", ["Chat", "Ingest sample"], index=0)
        if st.button("Clear chat"):
            st.session_state.pop("messages", None)
            st.rerun()

    st.title("Self-Evolving Knowledge Graph")
    st.caption(
        "Chat with the graph agent, or paste a real-world sample to ingest "
        "it as a distributed user contribution (guide §8)."
    )

    if gi.sleep_pass_running:
        st.warning("Sleep pass is running; chat is paused until it finishes.")

    st.session_state.setdefault("messages", [])

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if mode == "Chat":
        prompt = st.chat_input("Ask the graph, or give a management instruction…")
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
        return

    sample = st.chat_input("Paste a raw sample (paragraph) to ingest…")
    if not sample:
        return
    st.session_state.messages.append(
        {"role": "user", "content": f"**[Ingest]**\n\n{sample}"}
    )
    with st.chat_message("user"):
        st.markdown(f"**[Ingest]**\n\n{sample}")
    with st.chat_message("assistant"):
        with st.spinner("Extracting + integrating…"):
            try:
                ctx = _session_user_ctx()
                result = gi.integrate(
                    sample,
                    user_id=ctx.user_id,
                    conversation_id=ctx.conversation_id,
                    turn_id=ctx.turn_id,
                )
                reply = (
                    f"Ingested as **{result.classification}**: "
                    f"{result.nodes_touched} node(s), "
                    f"{result.edges_added} edge(s) added."
                )
                if result.compressed:
                    reply += (
                        f"\n\n> ℹ️ Input was {result.original_tokens} tokens "
                        f"(over the {INTEGRATE_INPUT_SOFT_CAP_TOKENS}-token soft cap); LLM-compressed to "
                        f"{result.compressed_tokens} tokens before extraction."
                    )
            except Exception as exc:
                reply = f"Ingest failed: {exc}"
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    gi.save()
    st.rerun()


if __name__ == "__main__":
    main()
