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

import re
import sys
import uuid
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

import streamlit as st
import streamlit.components.v1 as components

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
import time

from src.modules.m2_qa_agent import GraphAgent, fast_query, fast_query_stream

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
  /* Chat message body — larger text for readability (§16.14) */
  div[data-testid="stChatMessage"] p,
  div[data-testid="stChatMessage"] li,
  div[data-testid="stChatMessage"] div {
    font-size: 1.05rem !important;
    line-height: 1.6 !important;
  }
  /* Chat input textarea */
  div[data-testid="stChatInput"] textarea {
    font-size: 1.05rem !important;
  }
  /* Response timer (§16.15.4) — !important to win over the generic
     stChatMessage div rule above. Fixed px height matches the live
     iframe timer so the visual position doesn't shift on freeze. */
  .response-timer,
  div[data-testid="stChatMessage"] .response-timer {
    color: #888 !important;
    font-size: 14px !important;
    line-height: 24px !important;
    height: 24px !important;
    margin: 0 0 4px 0 !important;
    padding: 0 !important;
    font-variant-numeric: tabular-nums !important;
  }
  /* Animated status dots (cycle empty → "." → ".." → "..." every 4s) */
  div[data-testid="stChatMessage"] .agent-status,
  .agent-status {
    color: #888 !important;
    font-style: italic !important;
    font-size: 1rem !important;
    line-height: 1.5 !important;
  }
  /* Three discrete dot spans, each with its own visibility keyframe. */
  .agent-status .d {
    opacity: 0;
    display: inline !important;
  }
  .agent-status .d1 { animation: agent-blink1 4s infinite; }
  .agent-status .d2 { animation: agent-blink2 4s infinite; }
  .agent-status .d3 { animation: agent-blink3 4s infinite; }
  @keyframes agent-blink1 {
    0%, 24.9% { opacity: 0; }
    25%, 100% { opacity: 1; }
  }
  @keyframes agent-blink2 {
    0%, 49.9% { opacity: 0; }
    50%, 100% { opacity: 1; }
  }
  @keyframes agent-blink3 {
    0%, 74.9% { opacity: 0; }
    75%, 100% { opacity: 1; }
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


def _stream_internal_turn(
    gi: GraphInstance, prompt: str, prior_history: list[dict],
):
    """Streaming internal turn.  Yields ``{"status": ...}`` dicts for
    intermediate tool-loop steps and ``str`` tokens for the final reply."""
    if gi.sleep_pass_running:
        yield "Working-memory maintenance is running. Please try again in a moment."
        return
    agent = GraphAgent(gi)
    history = _windowed_history(prior_history)
    try:
        yield from agent.stream_call(prompt, history=history)
    except Exception as exc:
        yield f"Error: {exc}"


def _stream_external_turn(
    gi: GraphInstance, prompt: str, prior_history: list[dict],
):
    """Streaming external turn.  Yields scrubbed sentence chunks."""
    if gi.sleep_pass_running:
        yield "The service is briefly unavailable. Please try again in a moment."
        return
    try:
        history = _windowed_history(prior_history)
        yield from fast_query_stream(gi, prompt, mode="external", history=history)
    except Exception:
        yield "I don't have information on that."


_FROZEN_TIMER_TEMPLATE = """
<style>
    html, body {{ margin: 0; padding: 0; overflow: hidden; }}
    #rt {{
        color: #888;
        font-size: 14px;
        line-height: 24px;
        height: 24px;
        font-variant-numeric: tabular-nums;
        font-family: -apple-system, BlinkMacSystemFont,
            'Segoe UI', 'Helvetica Neue', sans-serif;
    }}
</style>
<div id="rt">⏱ {elapsed:.1f}s</div>
"""


def _render_frozen_timer(elapsed: float) -> None:
    """Render a static timer iframe with the same wrapper as the live one
    so the layout doesn't shift when the live iframe is replaced."""
    components.html(
        _FROZEN_TIMER_TEMPLATE.format(elapsed=elapsed),
        height=28,
    )


_INGEST_CMD_RE = re.compile(r"^\s*/ingest(?:\s+(.+))?\s*$", re.IGNORECASE)


_FILE_ICON_BY_EXT = {
    ".pdf": "📕",
    ".txt": "📄",
    ".md": "📝",
    ".markdown": "📝",
}


def _render_attachment_card(attachment: dict) -> None:
    """Render a ChatGPT-style file card inside the user's chat bubble.

    ``attachment`` is a small metadata dict — ``{"name", "size_kb",
    "ext"}`` — persisted in the message so the card survives reruns + chat
    reload. Bytes are NOT stored (they live only in session_state until
    committed and would bloat the chats DB)."""
    name = attachment.get("name", "(unknown)")
    size_kb = attachment.get("size_kb", 0)
    ext = attachment.get("ext", "").lower()
    icon = _FILE_ICON_BY_EXT.get(ext, "📎")
    label = ext.lstrip(".").upper() or "FILE"
    st.markdown(
        f'<div style="display: inline-flex; align-items: center; gap: 12px; '
        f'padding: 10px 14px; background: #2b2f36; color: #f0f0f0; '
        f'border-radius: 10px; border: 1px solid #3a3f47; '
        f'margin-bottom: 8px; max-width: 360px;">'
        f'<div style="font-size: 28px; line-height: 1;">{icon}</div>'
        f'<div style="display: flex; flex-direction: column; '
        f'min-width: 0; overflow: hidden;">'
        f'<div style="font-weight: 600; overflow: hidden; '
        f'text-overflow: ellipsis; white-space: nowrap;">{name}</div>'
        f'<div style="font-size: 12px; color: #a0a8b0;">'
        f'{label} · {size_kb} KB</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _commit_staged_upload(
    gi: GraphInstance,
    staged: dict,
    target_basename: str | None,
) -> str:
    """Write the in-memory staged file to data/raw/ then run /ingest.

    Implements the §16.13 staged-upload flow: drag drops the bytes into
    session state only; ``/ingest`` (any submission, with or without an
    argument) is what commits them to disk and triggers the ingest. The
    bytes always land at data/raw/<staged.name>; ``target_basename`` lets
    the user type ``/ingest <other>`` to ingest a different already-on-disk
    file while still persisting the attachment for later.
    """
    import unicodedata

    if gi.sleep_pass_running:
        return "Working-memory maintenance is running. Cannot ingest right now."

    staged_name = unicodedata.normalize("NFKC", staged["name"])
    raw_dir = Path("data") / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / staged_name
    dest.write_bytes(staged["bytes"])

    target = target_basename or staged_name
    agent = GraphAgent(gi)
    result = agent._ingest_file(target)

    if result.get("error"):
        return f"Upload failed: {result['error']}"

    nodes = result.get("nodes_added", 0)
    pass1_edges = result.get("edges_added", 0)
    pass2_edges = result.get("pass2_edges_added", 0)
    # Single-doc path uses `pages_processed`; super-chunk split path
    # exposes `total_pages` instead. Read both, prefer the split count
    # when present.
    pages = result.get("total_pages") or result.get("pages_processed", 0)
    fused = result.get("nodes_fused", 0)
    reclassified = result.get("edges_reclassified", 0)
    duplicates = result.get("duplicate_edges_removed", 0)
    superchunks = result.get("split_into_superchunks", 0)
    final_edges = pass1_edges + pass2_edges - duplicates
    parts = [f"Ingested **{target}**"]
    if pages > 1:
        page_label = f"({pages} pages"
        if superchunks > 1:
            ok = result.get("superchunks_ok", superchunks)
            page_label += f", split into {ok}/{superchunks} super-chunks"
        page_label += ")"
        parts.append(page_label)
    parts.append(
        f"— {nodes} nodes, {final_edges} edges "
        f"({pass2_edges} from PASS 2"
        + (f", {duplicates} dup removed" if duplicates else "")
        + ")"
        + (f", {reclassified} reclassified" if reclassified else "")
        + (f", {fused} fused" if fused else "")
        + "."
    )
    return " ".join(parts)


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
                if m["role"] == "assistant" and "elapsed" in m:
                    _render_frozen_timer(float(m["elapsed"]))
                if m["role"] == "user" and m.get("attachment"):
                    _render_attachment_card(m["attachment"])
                st.markdown(m.get("content") or "")

    # Persistent staged-file indicator (§16.13 follow-up). Streamlit's
    # chat_input clears its native file chip after submit, so once we move
    # the bytes into session state the user otherwise has no visible cue
    # that the upload is still pending. Render a chip above the input
    # with a × button so they can dismiss it without sending /ingest.
    if role == ROLE_INTERNAL and st.session_state.get("staged_upload"):
        staged = st.session_state["staged_upload"]
        size_kb = max(1, len(staged["bytes"]) // 1024)
        c1, c2 = st.columns([20, 1])
        with c1:
            st.markdown(
                f'<div style="padding: 8px 12px; background: #f0f4f8; '
                f'border-radius: 6px; border-left: 3px solid #4a90e2; '
                f'color: #333; margin-bottom: 6px;">'
                f'📎 <b>{staged["name"]}</b> ({size_kb} KB) staged — '
                f'type <code>/ingest</code> to save &amp; process.</div>',
                unsafe_allow_html=True,
            )
        with c2:
            if st.button("✕", key="clear_staged_upload",
                         help="Discard staged file"):
                st.session_state["staged_upload"] = None
                st.rerun()

    chat_kwargs: dict = {"placeholder": _placeholder_for(role)}
    if role == ROLE_INTERNAL:
        chat_kwargs["accept_file"] = True
        chat_kwargs["file_type"] = ["pdf", "txt", "md"]

    user_input = st.chat_input(**chat_kwargs)
    if not user_input:
        return

    uploaded_file = None
    if hasattr(user_input, "text"):
        uploaded_files = getattr(user_input, "files", None) or []
        uploaded_file = uploaded_files[0] if uploaded_files else None
        user_text = user_input.text or ""
    else:
        user_text = str(user_input)

    # §16.13 staged-upload flow: dragging a file stashes its bytes in
    # session state; ``/ingest`` (this submission or a later one) commits
    # them to data/raw/ and triggers the ingest pipeline.
    if uploaded_file and role == ROLE_INTERNAL:
        st.session_state["staged_upload"] = {
            "name": uploaded_file.name,
            "bytes": uploaded_file.read(),
        }

    ingest_match = _INGEST_CMD_RE.match(user_text)
    has_stage = bool(st.session_state.get("staged_upload"))

    # Path A — commit the staged file (file just attached or attached
    # earlier and the user is now typing /ingest). The ingest call is
    # synchronous and can take minutes on a large PDF; render the user
    # turn + a spinner BEFORE the blocking call so the page isn't frozen
    # on the pre-submit state while the backend works.
    if ingest_match and has_stage and role == ROLE_INTERNAL:
        staged = st.session_state["staged_upload"]
        arg = (ingest_match.group(1) or "").strip() or None
        display_text = user_text or "/ingest"
        attachment = {
            "name": staged["name"],
            "size_kb": max(1, len(staged["bytes"]) // 1024),
            "ext": Path(staged["name"]).suffix.lower(),
        }
        st.session_state["messages"].append({
            "role": "user",
            "content": display_text,
            "attachment": attachment,
        })
        with st.chat_message("user"):
            _render_attachment_card(attachment)
            st.markdown(display_text)
        with st.chat_message("assistant"):
            spinner_label = (
                f"Saving **{staged['name']}** to `data/raw/` and running "
                f"ingest. This can take a few minutes on a long PDF — "
                f"do not refresh the page."
            )
            with st.spinner(spinner_label):
                reply = _commit_staged_upload(gi, staged, arg)
            st.markdown(reply)
        st.session_state["staged_upload"] = None
        st.session_state["messages"].append({"role": "assistant", "content": reply})
        store.save_chat(
            active_conv_id, USER_ID_BY_ROLE[role], role, st.session_state["messages"],
        )
        gi.save()
        st.rerun()
        return

    # Path B — file attached this turn but no /ingest yet: silently stash
    # (the persistent indicator above the chat_input is the visual cue).
    # If the user also typed something non-empty that isn't /ingest, fall
    # through to the normal Q&A path so a typed question still gets
    # answered with the attachment held for later.
    if uploaded_file and role == ROLE_INTERNAL and not ingest_match and not user_text.strip():
        # Empty text + just attached: rerun to show the indicator. The
        # chat message log isn't touched — there's nothing to record yet.
        st.rerun()
        return

    prompt = user_text
    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        t0 = time.monotonic()

        timer_slot = st.empty()
        # Live JS timer in an iframe (components.html executes scripts).
        # Wrapped in st.empty() so we can replace it the moment the stream
        # ends — otherwise the iframe keeps ticking until st.rerun().
        with timer_slot:
            components.html(
                """
                <style>
                    html, body { margin: 0; padding: 0; overflow: hidden; }
                    #rt {
                        color: #888;
                        font-size: 14px;
                        line-height: 24px;
                        height: 24px;
                        font-variant-numeric: tabular-nums;
                        font-family: -apple-system, BlinkMacSystemFont,
                            'Segoe UI', 'Helvetica Neue', sans-serif;
                    }
                </style>
                <div id="rt">⏱ 0.0s</div>
                <script>
                (function() {
                    var s = Date.now();
                    var el = document.getElementById('rt');
                    setInterval(function() {
                        el.textContent = '⏱ ' + ((Date.now() - s) / 1000).toFixed(1) + 's';
                    }, 100);
                })();
                </script>
                """,
                height=28,
            )

        status_slot = st.empty()
        text_slot = st.empty()

        prior = st.session_state["messages"][:-1]
        if role == ROLE_INTERNAL:
            stream = _stream_internal_turn(gi, prompt, prior)
        else:
            stream = _stream_external_turn(gi, prompt, prior)

        collected: list[str] = []
        for chunk in stream:
            if isinstance(chunk, dict):
                if "status" in chunk:
                    # Strip trailing ellipsis / dots — CSS animates them.
                    label = chunk["status"].rstrip("…").rstrip(".")
                    status_slot.markdown(
                        f'<span class="agent-status">{label}'
                        f'<span class="d d1">.</span>'
                        f'<span class="d d2">.</span>'
                        f'<span class="d d3">.</span>'
                        f'</span>',
                        unsafe_allow_html=True,
                    )
            else:
                status_slot.empty()
                collected.append(chunk)
                text_slot.markdown("".join(collected))

        elapsed = time.monotonic() - t0
        # Freeze timer with another iframe (no script) so the surrounding
        # Streamlit wrapper stays consistent — swapping iframe → markdown
        # would shift vertical position.  Same wrapper as history render.
        with timer_slot:
            _render_frozen_timer(elapsed)
        status_slot.empty()
        reply = "".join(collected) or "(no response)"
        text_slot.markdown(reply)

    st.session_state["messages"].append(
        {"role": "assistant", "content": reply, "elapsed": round(elapsed, 1)},
    )
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
