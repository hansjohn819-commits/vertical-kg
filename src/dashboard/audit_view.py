"""Audit view — interactive knowledge graph (§16, 2026-05-06).

Single function `render_audit_view(gi)` that takes over the page when
the user clicks the Graph tab. Owns its own sidebar (history timeline)
and main area (cytoscape canvas). No role gating — visible to every
user once the tab is selected (per 2026-05-06 user request).

Dynamic refresh:
  - GraphInstance is process-cached so chat-side ingests + sleep passes
    mutate it in place; this view reads the latest state on every
    Streamlit rerun (which fires on tab switch / sidebar click).
  - 🔄 button forces an explicit rerun for users sitting on this tab
    while another browser tab triggers an ingest.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.dashboard.audit_render import (
    graph_to_cytoscape,
    parse_history_events,
    render_cytoscape_html,
)
from src.graph.instance import GraphInstance

_LOG_PATH = Path(__file__).resolve().parents[2] / "log.md"

# Streamlit's `components.html(height=N)` writes a fixed pixel height onto
# the iframe AND its wrapper. Several parent containers (block-container,
# element-container, stMain) also have their own height constraints that
# prevent the iframe from claiming the full viewport even when its own
# CSS says 100vh — the smallest ancestor wins. So we have to push
# `height: 100%` through every layer of the Streamlit DOM, then make the
# iframe itself fill 100% of its (now-unconstrained) parent. Combined with
# overflow:hidden at the root, the Graph tab becomes a fixed single-screen
# canvas, no body scroll. Chat tab is unaffected — this CSS only gets
# injected from inside `render_audit_view`.
_IFRAME_STRETCH_CSS = """
<style>
  /* Layer 0: app root locked to viewport, no scroll. */
  html, body { height: 100vh !important; overflow: hidden !important; }
  .stApp { height: 100vh !important; overflow: hidden !important; }

  /* Layer 1: main content area takes full app height. */
  section[data-testid="stMain"] {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
  }

  /* Layer 2: Streamlit's main block container (the inner padding wrapper).
     Trim padding so the iframe can claim more vertical real estate. */
  div[data-testid="stMainBlockContainer"],
  .block-container {
    height: 100% !important;
    max-height: 100% !important;
    overflow: hidden !important;
    padding-top: 0.5rem !important;
    padding-bottom: 0 !important;
  }

  /* Layer 3: every element-container / vertical-block above the iframe
     gets shrunk to its content; the LAST one (the iframe wrapper) gets
     flex:1 to absorb the remaining height. */
  div[data-testid="stVerticalBlock"] { height: 100% !important; }
  div[data-testid="stVerticalBlock"] > div:last-child { flex: 1 1 auto !important; }
  div[data-testid="stVerticalBlock"] > div:last-child .element-container { height: 100% !important; }

  /* Layer 4: the components.html wrapper. */
  div[data-testid="stCustomComponentV1"],
  div[data-testid="stIFrame"] {
    height: 100% !important;
    max-height: 100% !important;
  }

  /* Layer 5: the iframe itself fills its (now-correctly-sized) parent. */
  iframe[srcdoc] {
    height: 100% !important;
    min-height: 100% !important;
    width: 100% !important;
    border: none !important;
    display: block !important;
  }

  /* Hide leftover Streamlit chrome that can introduce vertical noise. */
  footer { display: none !important; }
  [data-testid="stStatusWidget"] { display: none !important; }
</style>
"""


def _build_view(
    instance: GraphInstance, selected_event: dict | None,
) -> tuple[dict, str]:
    """Return (cytoscape_elements, status_caption) for the current view."""
    storage = instance.storage
    if selected_event is None:
        elements = graph_to_cytoscape(storage)
        return elements, (
            f"Showing the full graph: {len(elements['nodes'])} nodes / "
            f"{len(elements['edges'])} edges"
        )

    if selected_event["kind"] == "ingest":
        rdoc = selected_event.get("raw_doc_id")
        focus_ids: set[str] = set()
        for n in storage.nodes():
            prov = getattr(n, "provenance", None)
            if prov and getattr(prov, "raw_doc_id", None) == rdoc:
                focus_ids.add(n.id)
        elements = graph_to_cytoscape(
            storage,
            focus_node_ids=focus_ids,
            highlight_node_ids=focus_ids,
            add_neighbor_hop=True,
        )
        return elements, (
            f"Ingest event focus: {len(focus_ids)} new nodes from "
            f"{rdoc} (+ 1-hop context)"
        )

    merged_ids = set(selected_event.get("merged_new_ids") or [])
    pruned_edge_ids = set(selected_event.get("pruned_edge_ids") or [])
    added_edge_ids = set(selected_event.get("added_link_edge_ids") or [])
    focus_ids = set(merged_ids)
    elements = graph_to_cytoscape(
        storage,
        focus_node_ids=focus_ids if focus_ids else None,
        highlight_node_ids=merged_ids,
        highlight_edge_ids=pruned_edge_ids | added_edge_ids,
        add_neighbor_hop=bool(focus_ids),
    )
    parts = []
    if merged_ids:
        parts.append(f"{len(merged_ids)} fused")
    if added_edge_ids:
        parts.append(f"{len(added_edge_ids)} new edges")
    if pruned_edge_ids:
        parts.append(f"{len(pruned_edge_ids)} pruned edges")
    return elements, (
        "Sleep pass focus: " + (" / ".join(parts) if parts else "no node-level changes recorded")
    )


def _format_local_ts(ts: str) -> str:
    """Convert an ISO 8601 UTC timestamp to local-time YYYY-MM-DD HH:MM
    for display. Server's local timezone (which equals the user's, since
    the dashboard is single-machine) is used. Falls back to a string
    split if parsing fails."""
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        if "T" in ts:
            return ts.split("T")[0] + " " + ts.split("T")[1][:5]
        return ts


def _render_history_sidebar(events: list[dict]) -> dict | None:
    with st.sidebar:
        st.markdown("### Knowledge Graph")
        if st.button("🔄 Refresh", use_container_width=True,
                     help="Re-read the graph and event log."):
            st.rerun()

        st.divider()
        active_id = st.session_state.get("audit_event_id", "__now__")
        now_active = active_id == "__now__"
        # The "live current state" entry — primary-styled when active so
        # it stands out from the historical event list below.
        if st.button(
            "🟢 Knowledge Graph (live)" if now_active else "Knowledge Graph (live)",
            use_container_width=True,
            type="primary" if now_active else "secondary",
            key="audit_now_btn",
        ):
            st.session_state["audit_event_id"] = "__now__"
            st.rerun()

        st.markdown("###### Activity log")
        if not events:
            st.caption("No events yet.")
            return None

        for ev in events:
            eid = ev["id"]
            is_active = (eid == active_id)
            ts = ev.get("ts", "")
            ts_short = _format_local_ts(ts)
            help_text = ts + "\n" + ev["summary"]
            if st.button(
                ("🟢 " if is_active else "") + ev["summary"],
                use_container_width=True,
                key=f"audit_ev_{eid}",
                help=help_text,
            ):
                st.session_state["audit_event_id"] = eid
                st.rerun()
            st.caption(ts_short)

    if active_id and active_id != "__now__":
        for ev in events:
            if ev["id"] == active_id:
                return ev
    return None


def render_audit_view(gi: GraphInstance) -> None:
    """Render the full Graph tab — sidebar history + cytoscape canvas."""
    events = parse_history_events(_LOG_PATH)
    selected = _render_history_sidebar(events)

    if gi.sleep_pass_running:
        st.warning(
            "Working-memory maintenance is in progress — the graph may shift "
            "while operations complete. Click 🔄 Refresh in the sidebar to "
            "see the latest state."
        )

    elements, status = _build_view(gi, selected)
    st.caption(status)

    # Stretch-to-viewport CSS injection (one-shot per render is fine — Streamlit
    # de-dupes identical <style> blocks during a rerun). The fallback `height`
    # passed to components.html below is a placeholder; the real sizing comes
    # from this CSS override.
    st.markdown(_IFRAME_STRETCH_CSS, unsafe_allow_html=True)

    html_str = render_cytoscape_html(elements)
    components.html(html_str, height=900, scrolling=False)
