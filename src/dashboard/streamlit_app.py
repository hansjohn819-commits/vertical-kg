"""Streamlit entry — top-tab nav + per-tab render dispatch.

Three tabs across the top, left to right:
  1. Dashboard — placeholder for a future big-screen overview dashboard
                 (per 2026-05-06 user request: button is reserved but
                 has no behaviour yet).
  2. Q&A       — chat with the kelp-industry analyst (internal/external
                 role inside, original behaviour preserved).
  3. Graph     — Obsidian-style interactive knowledge-graph viewer.

Streamlit's default `pages/` sidebar nav is hidden via CSS; navigation
is fully driven by the top buttons + session_state["active_tab"]. Each
tab function owns its own sidebar content during render.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

import streamlit as st

from src.dashboard.audit_view import render_audit_view
from src.dashboard.chat_store import (
    EXTERNAL_DEFAULT_USER_ID,
    INTERNAL_DEFAULT_USER_ID,
    ROLE_EXTERNAL,
    ROLE_INTERNAL,
    ChatStore,
)
from src.graph.instance import GraphInstance
from src.graph.tokens import count_tokens
from src.modules.m2_qa_agent import GraphAgent, fast_query

WORKSPACE = Path(__file__).resolve().parents[2]
INSTANCE_DIR = WORKSPACE / "data" / "production"
ONTOLOGY_PATH = WORKSPACE / "ontology.md"
CHATS_DB_PATH = WORKSPACE / "data" / "chats.db"

HISTORY_BUDGET_TOKENS = 3_000  # guide §12.5.2 chat-history allotment

USER_ID_BY_ROLE = {
    ROLE_INTERNAL: INTERNAL_DEFAULT_USER_ID,
    ROLE_EXTERNAL: EXTERNAL_DEFAULT_USER_ID,
}

TABS = ["Dashboard", "Q&A", "Graph"]
DEFAULT_TAB = "Q&A"


# --- Cached resources ------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_instance() -> GraphInstance:
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    return GraphInstance(
        name="production", storage_path=INSTANCE_DIR, ontology_path=ONTOLOGY_PATH,
    )


@st.cache_resource(show_spinner=False)
def _load_chat_store() -> ChatStore:
    return ChatStore(CHATS_DB_PATH)


# --- Top tab nav -----------------------------------------------------------

_HIDE_DEFAULT_NAV_CSS = """
<style>
  /* Hide Streamlit's default top header (empty dark bar with menu/deploy) — */
  /* our 3 nav buttons sit at the visual top instead. */
  header[data-testid="stHeader"] { display: none !important; }
  /* And the default sidebar pages-nav (we have a custom top nav). */
  [data-testid="stSidebarNav"] { display: none !important; }
  /* Pull main content up to where the header used to be. */
  .stApp > div:first-child { margin-top: 0 !important; }
  .block-container {
    padding-top: 0.6rem !important;
    padding-bottom: 1rem;
  }
  /* Snug the nav buttons against the top edge of the viewport. */
  div[data-testid="stHorizontalBlock"]:first-of-type {
    margin-top: 0;
  }
  /* Slightly larger / weightier nav buttons since they're acting as a tab bar. */
  div[data-testid="stHorizontalBlock"]:first-of-type button[kind] {
    font-weight: 600;
    letter-spacing: 0.02em;
  }
</style>
"""


def _render_top_nav() -> str:
    """Render 3 buttons across the top. Active tab gets primary styling.
    Returns the active tab name."""
    st.markdown(_HIDE_DEFAULT_NAV_CSS, unsafe_allow_html=True)

    active = st.session_state.get("active_tab", DEFAULT_TAB)
    if active not in TABS:
        active = DEFAULT_TAB

    cols = st.columns([1, 1, 1, 8])  # 3 nav cells + spacer
    for i, tab in enumerate(TABS):
        is_active = tab == active
        if cols[i].button(
            tab,
            use_container_width=True,
            type=("primary" if is_active else "secondary"),
            key=f"navbtn_{tab}",
        ):
            if tab != active:
                st.session_state["active_tab"] = tab
                st.rerun()
    st.divider()
    return active


# --- Tab: Dashboard (placeholder) ------------------------------------------

def _render_dashboard_tab() -> None:
    st.title("Dashboard")
    st.markdown(
        "An at-a-glance overview dashboard will live here — KPIs, recent "
        "activity, and big-screen visuals tailored to the kelp / seaweed "
        "industry. Reserved space for upcoming work."
    )
    st.info(
        "🚧 Coming soon. Use **Q&A** to talk to the analyst or **Graph** "
        "to explore the knowledge graph in the meantime."
    )


# --- Tab: Q&A (chat) -------------------------------------------------------

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


def _start_fresh_pending() -> None:
    st.session_state["active_conv_id"] = str(uuid.uuid4())
    st.session_state["messages"] = []
    st.session_state["is_pending"] = True


def _ensure_session_initialized() -> None:
    if "active_conv_id" not in st.session_state:
        _start_fresh_pending()
    if "prev_role" not in st.session_state:
        st.session_state["prev_role"] = st.session_state.get("role_select", ROLE_EXTERNAL)


def _react_to_role_change(current_role: str) -> None:
    if current_role != st.session_state.get("prev_role"):
        _start_fresh_pending()
        st.session_state["prev_role"] = current_role


def _switch_to_chat(store: ChatStore, conv_id: str) -> None:
    _meta, msgs = store.load_chat(conv_id)
    st.session_state["active_conv_id"] = conv_id
    st.session_state["messages"] = msgs
    st.session_state["is_pending"] = False


def _delete_chat(store: ChatStore, conv_id: str) -> None:
    store.delete_chat(conv_id)
    if st.session_state.get("active_conv_id") == conv_id:
        _start_fresh_pending()


def _render_chat_sidebar(store: ChatStore, role: str) -> None:
    with st.sidebar:
        st.markdown("### Q&A")
        st.selectbox(
            "Role",
            options=[ROLE_INTERNAL, ROLE_EXTERNAL],
            index=0 if role == ROLE_INTERNAL else 1,
            key="role_select",
            help=(
                "Internal: full research mode with knowledge-base maintenance. "
                "External: fast read-only analyst replies."
            ),
        )
        st.divider()
        if st.button("➕ New chat", use_container_width=True):
            _start_fresh_pending()
            st.rerun()

        user_id = USER_ID_BY_ROLE[role]
        chats = store.list_chats(user_id)
        active = st.session_state.get("active_conv_id")
        is_pending = st.session_state.get("is_pending", True)

        st.caption(f"{len(chats)} conversation{'s' if len(chats) != 1 else ''}")
        for chat in chats:
            label = chat["name"] or "(untitled)"
            is_active = (chat["id"] == active) and not is_pending
            cols = st.columns([0.85, 0.15])
            with cols[0]:
                if st.button(
                    f"{'🟢 ' if is_active else ''}{label}",
                    key=f"chat_open_{chat['id']}",
                    use_container_width=True,
                ):
                    _switch_to_chat(store, chat["id"])
                    st.rerun()
            with cols[1]:
                if st.button("🗑", key=f"chat_del_{chat['id']}", help="Delete"):
                    _delete_chat(store, chat["id"])
                    st.rerun()


def _render_chat_header(role: str, sleep_pass_running: bool) -> None:
    if role == ROLE_INTERNAL:
        st.title("Seaweed Industry Insights — Internal")
        st.caption(
            "Research mode. Ask about the kelp / seaweed industry, drop a "
            "new report into data/raw/ and ask the analyst to ingest it, "
            "or trigger maintenance operations on the working memory."
        )
        if sleep_pass_running:
            st.warning(
                "Working-memory maintenance is running; chat is paused "
                "until it finishes."
            )
    else:
        st.title("Seaweed Industry Insights")
        st.caption(
            "Ask about the kelp / seaweed industry — trends, economics, "
            "players, ecology, technology."
        )


def _placeholder_for(role: str) -> str:
    if role == ROLE_INTERNAL:
        return "Ask about the kelp industry, or ask to ingest a report…"
    return "Ask about the kelp industry…"


def _handle_internal_turn(
    gi: GraphInstance, prompt: str, prior_history: list[dict],
) -> str:
    if gi.sleep_pass_running:
        return (
            "Working-memory maintenance is running. Please try again "
            "in a moment."
        )
    agent = GraphAgent(gi)
    history = _windowed_history(prior_history)
    try:
        return agent.call(prompt, history=history)
    except Exception as exc:
        return f"Error: {exc}"


def _handle_external_turn(
    gi: GraphInstance, prompt: str, prior_history: list[dict],
) -> str:
    if gi.sleep_pass_running:
        return "The service is briefly unavailable. Please try again in a moment."
    try:
        history = _windowed_history(prior_history)
        return fast_query(gi, prompt, mode="external", history=history)
    except Exception:
        return "I don't have information on that."


def _render_qa_tab(gi: GraphInstance, store: ChatStore) -> None:
    _ensure_session_initialized()
    role = st.session_state.get("role_select", ROLE_EXTERNAL)
    _react_to_role_change(role)

    _render_chat_sidebar(store, role)
    _render_chat_header(role, gi.sleep_pass_running)

    active_conv_id = st.session_state["active_conv_id"]

    for m in st.session_state.get("messages", []):
        if m["role"] in ("user", "assistant"):
            with st.chat_message(m["role"]):
                st.markdown(m.get("content") or "")

    user_input = st.chat_input(_placeholder_for(role))
    if not user_input:
        return

    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…" if role == ROLE_INTERNAL else "Looking it up…"):
            if role == ROLE_INTERNAL:
                reply = _handle_internal_turn(
                    gi, user_input, st.session_state["messages"][:-1],
                )
            else:
                reply = _handle_external_turn(
                    gi, user_input, st.session_state["messages"][:-1],
                )
        st.markdown(reply)

    st.session_state["messages"].append({"role": "assistant", "content": reply})
    store.save_chat(
        active_conv_id, USER_ID_BY_ROLE[role], role, st.session_state["messages"],
    )
    st.session_state["is_pending"] = False
    if role == ROLE_INTERNAL:
        gi.save()
    st.rerun()


# --- Main ------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Seaweed Industry Insights", layout="wide")
    gi = _load_instance()
    store = _load_chat_store()

    active = _render_top_nav()

    if active == "Dashboard":
        _render_dashboard_tab()
    elif active == "Graph":
        render_audit_view(gi)
    else:  # Q&A (default)
        _render_qa_tab(gi, store)


if __name__ == "__main__":
    main()
