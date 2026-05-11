"""Audit-page rendering helpers (§16 audit page, 2026-05-06).

Three responsibilities:
  1. Slice a `GraphStorage` into a Cytoscape.js `elements` payload, with
     optional focus/highlight semantics (subgraph view for past events).
  2. Parse `log.md` into a deduplicated event timeline (sleep pass +
     M1 ingest), grouped + sorted for the sidebar.
  3. Render a self-contained HTML page that runs Cytoscape.js with the
     fcose layout, Obsidian-like dark styling, and a floating-card
     overlay that hovers near the clicked node.

Communication is one-way (Python → HTML). Click events are handled
entirely in the embedded JS; we don't roundtrip selections back to
Python because the floating-card UX is purely browser-side.
"""

from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.graph.retrieval import display_title
from src.graph.storage import GraphStorage

# --- Node type → 5-color palette (Obsidian-like dark) -----------------------
# Five buckets; everything that doesn't match goes to NEUTRAL. Names are
# checked case-insensitive against the bucket keyword set.
TYPE_BUCKETS: list[tuple[str, set[str]]] = [
    ("org",     {"organization", "business", "university", "producer",
                  "company", "agency", "association", "cooperative",
                  "publisher", "author", "group"}),
    ("person",  {"person"}),
    ("place",   {"location", "region", "country", "place", "site"}),
    ("concept", {"industry", "sector", "subsector", "concept", "strategy",
                  "strategic priority", "domain", "field", "topic",
                  "category", "framework", "goal", "market segment",
                  "agreement", "publication", "report", "study"}),
]
BUCKET_COLOR = {
    "org":     "#f7768e",   # warm red (Obsidian accent)
    "person":  "#7aa2f7",   # blue
    "place":   "#9ece6a",   # green
    "concept": "#bb9af7",   # purple
    "neutral": "#7d8590",   # cool gray (everything else)
}


def _bucket_for_type(node_type: str) -> str:
    nt = (node_type or "").strip().lower()
    for bucket, keywords in TYPE_BUCKETS:
        if nt in keywords:
            return bucket
    return "neutral"


# --- Event parsing ----------------------------------------------------------

_LOG_LINE_RE = re.compile(r"^- \[(?P<ts>[^\]]+)\] (?P<head>.*?) \| (?P<json>\{.*\})$")


def _parse_log_lines(log_path: Path) -> list[dict]:
    """Yield one parsed dict per log.md line. Robust to malformed lines —
    they're silently skipped (best-effort)."""
    if not log_path.exists():
        return []
    out: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = _LOG_LINE_RE.match(line.strip())
        if not m:
            continue
        try:
            payload = json.loads(m.group("json"))
        except json.JSONDecodeError:
            continue
        payload.setdefault("ts", m.group("ts"))
        out.append(payload)
    return out


def parse_history_events(log_path: Path) -> list[dict]:
    """Return audit-page events newest-first.

    Sleep passes are aggregated by `pass_id` — every event sharing a
    pass_id collapses to a single timeline entry that summarizes counts
    (merges, edges_pruned, nodes_pruned, new_links). M1 ingests are
    individual entries keyed by `run_id`. Failed sleep passes (no
    `pass_end`) still appear, marked with `terminated=True`.

    Each returned event dict has the shape:
      {
        "id": str,             # pass_id or run_id (stable, used as key)
        "kind": "pass" | "ingest",
        "ts": ISO 8601,
        "summary": str,        # one-line for sidebar display
        "details": dict,       # full counters / fields for tooltip / detail view
        "raw_doc_id": str | None,  # ingest only — used for subgraph slicing
        "merged_new_ids": list[str],  # pass only — for subgraph slicing
        "pruned_edge_ids": list[str], # pass only
        "added_link_edge_ids": list[str], # pass only
      }
    """
    events = _parse_log_lines(log_path)

    # Group sleep-pass events by pass_id
    by_pass: dict[str, dict] = defaultdict(lambda: {
        "kind": "pass",
        "ts": None,
        "id": None,
        "instance": None,
        "merges": 0, "edges_pruned": 0, "nodes_pruned": 0,
        "new_links": 0, "reinforced": 0,
        "merged_new_ids": [],
        "pruned_edge_ids": [],
        "added_link_edge_ids": [],
        "terminated": True,   # flipped to False if pass_end seen
        "details_lines": [],
    })
    ingests: list[dict] = []

    for ev in events:
        kind = ev.get("kind")
        ts = ev.get("ts")
        pid = ev.get("pass_id")

        if kind == "ingest_done":
            rdoc = ev.get("raw_doc_id") or ""
            from src.graph.retrieval import display_title as _dt
            human_title = _dt(rdoc) if rdoc else rdoc
            n_nodes = ev.get("nodes_added") or 0
            n_edges = ev.get("edges_added") or 0
            ingests.append({
                "id": ev.get("run_id") or f"ingest-{ts}",
                "kind": "ingest",
                "ts": ts,
                "summary": (
                    f"Knowledge ingestion — {human_title} "
                    f"(+{n_nodes} nodes, +{n_edges} edges)"
                ),
                "raw_doc_id": rdoc,
                "details": {
                    "raw_doc_id": rdoc,
                    "nodes_added": ev.get("nodes_added"),
                    "edges_added": ev.get("edges_added"),
                    "edges_skipped": ev.get("edges_skipped"),
                    "extraction_warning": ev.get("extraction_warning"),
                    "finish_reason": ev.get("finish_reason"),
                    "run_id": ev.get("run_id"),
                },
            })
            continue

        if not pid:
            continue
        bucket = by_pass[pid]
        bucket["id"] = pid
        # First-seen ts wins; updated to latest as we walk forward
        if not bucket["ts"]:
            bucket["ts"] = ts
        bucket["latest_ts"] = ts
        if kind == "pass_start":
            bucket["instance"] = ev.get("summary", "").replace("instance=", "").strip()
        elif kind == "merge":
            bucket["merges"] += 1
            new_id = ev.get("new_id")
            if new_id:
                bucket["merged_new_ids"].append(new_id)
            bucket["details_lines"].append(
                f"merge: {ev.get('summary','')[:140]}"
            )
        elif kind == "merge_caveat":
            bucket["details_lines"].append(
                f"caveat: {ev.get('summary','')[:140]}"
            )
        elif kind == "prune":
            edge_id = ev.get("edge_id")
            if edge_id:
                bucket["pruned_edge_ids"].append(edge_id)
                bucket["edges_pruned"] += 1
            else:
                bucket["nodes_pruned"] += 1
            bucket["details_lines"].append(
                f"prune: {ev.get('summary','')[:140]}"
            )
        elif kind == "link_form":
            bucket["new_links"] += 1
            edge_id = ev.get("edge_id")
            if edge_id:
                bucket["added_link_edge_ids"].append(edge_id)
            bucket["details_lines"].append(
                f"link: {ev.get('summary','')[:140]}"
            )
        elif kind == "reinforce":
            bucket["reinforced"] += 1
        elif kind == "pass_end":
            bucket["terminated"] = False
            bucket["latest_ts"] = ts
            # Use returned counters if richer than rolled-up version
            for f in ("merges", "edges_pruned", "nodes_pruned",
                      "new_links", "reinforced"):
                if isinstance(ev.get(f), int):
                    bucket[f] = ev[f]

    pass_events: list[dict] = []
    for pid, b in by_pass.items():
        bits = []
        if b["merges"]:
            bits.append(f"{b['merges']} merges")
        if b["nodes_pruned"]:
            bits.append(f"{b['nodes_pruned']} nodes pruned")
        if b["edges_pruned"]:
            bits.append(f"{b['edges_pruned']} edges pruned")
        if b["new_links"]:
            bits.append(f"{b['new_links']} new links")
        if b["terminated"]:
            bits.append("terminated")
        summary = "Maintenance — " + (", ".join(bits) if bits else "no changes")
        pass_events.append({
            "id": pid,
            "kind": "pass",
            "ts": b["ts"],
            "summary": summary,
            "raw_doc_id": None,
            "merged_new_ids": b["merged_new_ids"],
            "pruned_edge_ids": b["pruned_edge_ids"],
            "added_link_edge_ids": b["added_link_edge_ids"],
            "details": {
                "instance": b.get("instance"),
                "merges": b["merges"],
                "edges_pruned": b["edges_pruned"],
                "nodes_pruned": b["nodes_pruned"],
                "new_links": b["new_links"],
                "reinforced": b["reinforced"],
                "terminated": b["terminated"],
                "lines": b["details_lines"][:50],  # cap detail
            },
        })

    all_events = ingests + pass_events
    # Sort newest first (ISO 8601 sorts lexically)
    all_events.sort(key=lambda e: e.get("ts") or "", reverse=True)
    return all_events


# --- Graph → cytoscape JSON -------------------------------------------------

def graph_to_cytoscape(
    storage: GraphStorage,
    *,
    focus_node_ids: set[str] | None = None,
    highlight_node_ids: set[str] | None = None,
    highlight_edge_ids: set[str] | None = None,
    add_neighbor_hop: bool = False,
    layout_positions: dict[str, tuple[float, float]] | None = None,
) -> dict:
    """Build {nodes: [...], edges: [...], stats: {...}} for cytoscape.

    `focus_node_ids` — if provided, only emit these nodes (+ optional 1-hop
                       neighbors when `add_neighbor_hop`). Used by past-event
                       focused-subgraph view.
    `highlight_node_ids` / `highlight_edge_ids` — visually flagged
                       (CSS class `highlight`) but rendered the same way.
    `layout_positions` — task 3 (2026-05-11): precomputed {node_id: (x, y)}
                       attached to each node so the frontend uses
                       ``layout: 'preset'`` (zero browser compute) instead
                       of running fcose. Nodes missing from the map fall
                       back to a tiny random offset around origin so a
                       just-ingested-but-not-yet-laid-out node still
                       renders.

    Active nodes only (skip ghosts where merged_into is set), unless the
    ghost is explicitly in focus_node_ids (so a past merge event can show
    the now-retired originals).

    Returns dict with three keys:
      ``nodes`` / ``edges`` — cytoscape elements payload.
      ``stats``             — {total_nodes, total_edges, orphans_hidden,
                               ghosts_hidden} computed against the full
                               storage so the caption can show authoritative
                               counts even when filtered.
    """
    focus = focus_node_ids
    if focus and add_neighbor_hop:
        expanded: set[str] = set(focus)
        for nid in list(focus):
            for nb in storage.neighbors(nid):
                expanded.add(nb.id)
        focus = expanded

    # Stats are computed against the WHOLE graph (regardless of focus) so
    # the caption reports the authoritative total, not the filtered view.
    total_nodes = 0
    total_edges = 0
    orphans_hidden = 0
    ghosts_hidden = 0
    for n in storage.nodes():
        total_nodes += 1
        if n.merged_into is not None:
            ghosts_hidden += 1
        elif storage.degree(n.id) == 0:
            orphans_hidden += 1
    for _ in storage.edges():
        total_edges += 1

    nodes_out: list[dict] = []
    seen_node_ids: set[str] = set()
    import random as _random
    _rng = _random.Random(0)  # deterministic fallback positions
    for n in storage.nodes():
        if focus is not None and n.id not in focus:
            continue
        if focus is None:
            if n.merged_into is not None:
                continue   # default view excludes ghosts
            if storage.degree(n.id) == 0:
                continue   # default view excludes orphan nodes (M1 extracted
                           # the entity but never linked it; see 2026-05-06
                           # bug report — orphans cluster as a fcose blob).
                           # Focus views still show them when explicitly
                           # included via focus_node_ids.
        bucket = _bucket_for_type(n.type)
        prov = getattr(n, "provenance", None)
        rid = getattr(prov, "raw_doc_id", None) if prov is not None else None
        source_quote = getattr(prov, "line_or_span", None) if prov is not None else None
        title = display_title(rid) if rid else None
        degree = storage.degree(n.id)
        elem: dict = {
            "data": {
                "id": n.id,
                "label": n.label,
                "type": n.type,
                "summary": n.summary or "",
                "source_title": title or "",
                "source_quote": source_quote or "",
                "raw_doc_id": rid or "",
                "version": n.version,
                "weight": round(n.weight, 3),
                "bucket": bucket,
                "color": BUCKET_COLOR[bucket],
                "highlight": bool(highlight_node_ids and n.id in highlight_node_ids),
                "is_ghost": n.merged_into is not None,
                "merged_into": n.merged_into or "",
                "detail": (n.detail or "")[:1500],
                "degree": degree,
                # Pre-scaled size dimensions for cytoscape stylesheet
                # `data()` references. Seeded at scale=1.0; JS updates on
                # zoom by multiplying the BASE values by the counter-scale
                # factor (1/zoom above FREEZE_ZOOM, else 1.0). Using
                # `data()` instead of `mapData()` because cytoscape's
                # mapData doesn't reliably re-evaluate on data changes —
                # plain data() refs do.
                "cs_w": 22,        # base width/height
                "cs_w_hl": 28,     # highlight (focus subgraph) size
                "cs_font": 11,     # label font
                "cs_b": 1.5,       # base border thickness
                "cs_b_sel": 3.5,   # :selected border thickness
                "cs_b_hl": 3,      # [?highlight] border thickness
                "cs_outline": 2,   # text outline width (label halo)
                "cs_tmy": -6,      # label vertical margin (above node)
            },
        }
        if layout_positions is not None:
            xy = layout_positions.get(n.id)
            if xy is None:
                # Just-ingested node not in cache yet — random offset near
                # origin keeps the preset layout working until the next save
                # refreshes the cache.
                elem["position"] = {
                    "x": _rng.uniform(-50, 50),
                    "y": _rng.uniform(-50, 50),
                }
            else:
                elem["position"] = {"x": xy[0], "y": xy[1]}
        nodes_out.append(elem)
        seen_node_ids.add(n.id)

    edges_out: list[dict] = []
    for e in storage.edges():
        if e.source_id not in seen_node_ids or e.target_id not in seen_node_ids:
            continue
        edges_out.append({
            "data": {
                "id": e.id,
                "source": e.source_id,
                "target": e.target_id,
                "label": e.type,
                "weight": round(e.weight, 3),
                "highlight": bool(highlight_edge_ids and e.id in highlight_edge_ids),
                # Pre-scaled edge dimensions (mirror of node cs_w family).
                "cs_ew": 1.1,       # base line width
                "cs_ew_sel": 2.2,   # :selected line width
                "cs_ew_hl": 2.0,    # [?highlight] line width
                "cs_arrow": 0.7,    # arrow-head scale
            },
        })

    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "stats": {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "orphans_hidden": orphans_hidden,
            "ghosts_hidden": ghosts_hidden,
            "rendered_nodes": len(nodes_out),
            "rendered_edges": len(edges_out),
        },
    }


# --- HTML template ----------------------------------------------------------

def _empty_canvas_html(message: str) -> str:
    """Tiny standalone page used when there's nothing to render."""
    msg = html.escape(message)
    return f"""
<!doctype html><html><head><meta charset="utf-8"><style>
  html,body{{margin:0;padding:0;width:100%;height:100%;background:#1a1b26;
            color:#7d8590;
            font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}}
  .empty{{height:100%;display:flex;align-items:center;justify-content:center;
          font-size:14px;letter-spacing:0.02em;}}
</style></head><body><div class="empty">{msg}</div></body></html>
"""


def render_cytoscape_html(
    elements: dict,
    *,
    empty_message: str = "No knowledge yet — ingest a document via chat to populate the graph.",
) -> str:
    """Self-contained HTML page rendering elements with cytoscape.js.

    Sizing: inner body + #cy use 100% of the iframe's content area, which
    is itself stretched to fill the viewport via CSS injected by the
    parent page (see `audit_view._inject_iframe_stretch_css`). So the
    canvas naturally tracks browser-window height with no hard-coded px.

    Behaviour:
      - Preset layout (positions precomputed in Python; zero browser cost)
      - Infinite-ish zoom (0.05 ≤ zoom ≤ 100)
      - Viewport degree-threshold: at most VIEWPORT_NODE_CAP nodes visible
        at any zoom level; threshold T rises as more nodes enter viewport
        and falls as the user zooms into a sub-region (task 3 B, 2026-05-11)
      - Node click → floating card near node, smooth fade
      - Empty space click → card fades out
      - Other-node click → card jumps to new position with new content
      - Highlighted nodes / edges get glow accent
    """
    if not elements.get("nodes"):
        return _empty_canvas_html(empty_message)

    payload = json.dumps(elements, ensure_ascii=False)

    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body {{
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    /* Gradient lives on body so particle canvas + cytoscape canvas can
       both stay transparent and let it show through. */
    background: radial-gradient(ellipse at center, #1f2235 0%, #16171f 100%);
    color: #c0caf5;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 "PingFang SC", "Microsoft YaHei", sans-serif;
    overflow: hidden;
    position: relative;
  }}
  #bg-particles {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 0;
    pointer-events: none;
    display: block;
  }}
  #cy {{
    width: 100%;
    height: 100%;
    background: transparent;
    position: relative;
    z-index: 1;
  }}
  /* Floating detail card */
  #card {{
    position: absolute;
    pointer-events: auto;
    min-width: 240px;
    max-width: 360px;
    padding: 14px 16px;
    border-radius: 10px;
    background: rgba(36, 40, 59, 0.96);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(122, 162, 247, 0.35);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
    color: #c0caf5;
    font-size: 12.5px;
    line-height: 1.5;
    opacity: 0;
    transform: translateY(4px);
    transition: opacity 0.18s ease, transform 0.18s ease;
    z-index: 10;
    word-break: break-word;
  }}
  #card.visible {{
    opacity: 1;
    transform: translateY(0);
  }}
  #card .card-type {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(122, 162, 247, 0.18);
    color: #7aa2f7;
    font-size: 11px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }}
  #card .card-label {{
    font-size: 15px;
    font-weight: 600;
    color: #e0e6ff;
    margin-bottom: 8px;
    line-height: 1.3;
  }}
  #card .card-summary {{
    color: #a9b1d6;
    margin-bottom: 8px;
  }}
  #card .card-source {{
    color: #7d8590;
    font-style: italic;
    font-size: 11.5px;
  }}
  #card .card-source-label {{
    color: #565f89;
    font-style: normal;
  }}
  #card .more-toggle {{
    margin-top: 10px;
    color: #7aa2f7;
    cursor: pointer;
    font-size: 11.5px;
    user-select: none;
  }}
  #card .more-toggle:hover {{
    text-decoration: underline;
  }}
  #card .more-content {{
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid rgba(122, 162, 247, 0.18);
    display: none;
  }}
  #card .more-content.visible {{
    display: block;
  }}
  #card .more-content .field {{
    margin-bottom: 6px;
  }}
  #card .more-content .field-name {{
    color: #565f89;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  #card .more-content .field-value {{
    color: #c0caf5;
    word-break: break-all;
  }}
  #card .more-content .quote {{
    color: #9ece6a;
    font-style: italic;
    font-size: 11.5px;
    border-left: 2px solid #9ece6a;
    padding-left: 8px;
    margin-top: 4px;
  }}
</style>
<script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
</head>
<body>
<canvas id="bg-particles"></canvas>
<div id="cy"></div>
<div id="card"></div>
<script>
  /* ---- Background particle drift (Obsidian-style ambient motion) ----
     Lightweight HTML5 canvas, slow-drifting particles + faint inter-
     connections. Fully behind cytoscape (z-index 0 vs 1) — graph
     interaction unaffected.

     Sizing: at script load time the iframe layout hasn't settled, so
     bg.offsetWidth/Height can be 0. We defer particle creation /
     scatter until the canvas reports a real size, and re-scatter on
     resize so they spread over the whole canvas after layout snaps. */
  (function startParticles() {{
    const bg = document.getElementById('bg-particles');
    const ctx = bg.getContext('2d');
    const COLORS = [
      'rgba(122, 162, 247, ',  // blue (primary)
      'rgba(187, 154, 247, ',  // purple
      'rgba(158, 206, 106, ',  // green (rare)
    ];

    let parts = [];
    let lastW = 0, lastH = 0;

    function makeParticle(w, h) {{
      // Pseudo-3D depth z ∈ [0..1]: 0 = far (small/slow/dim),
      // 1 = near (big/fast/bright/glowing). Most particles bias toward
      // far via z = rand^1.6 — sea of tiny faint background particles
      // plus a few prominent foreground ones, "deep space" feel.
      const z = Math.pow(Math.random(), 1.6);
      const speed = 0.06 + z * 0.32;
      const radius = 2.5 + z * z * 9.0;        // 2.5 (far) → ~11.5 (near)
      const alpha = 0.18 + z * 0.50;
      return {{
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * speed,
        vy: (Math.random() - 0.5) * speed,
        r: radius,
        a0: alpha,
        z: z,
        phase: Math.random() * Math.PI * 2,
        // Independent rotation around X and Y axes for organic tumbling.
        // Initial rotX biased toward [0.2, 1.0] rad so particles never start
        // perfectly head-on (which would read as a flat triangle for one
        // frame) — purely cosmetic. Both speeds independent random.
        rotX: 0.2 + Math.random() * 0.8,
        rotY: Math.random() * Math.PI * 2,
        rotXSpeed: (Math.random() - 0.5) * 0.018,
        rotYSpeed: (Math.random() - 0.5) * 0.022,
        color: COLORS[Math.random() < 0.85 ? 0 : (Math.random() < 0.7 ? 1 : 2)],
      }};
    }}

    /* Regular tetrahedron renderer — real 3D vertices, projected to 2D,
       with back-face culling and painter's algorithm. Spinning around the
       vertical axis with a fixed X-axis tilt for a consistent 3/4 view.
       Lambert-style shading by face normal · light direction. */

    // 4 vertices of a regular tetrahedron, apex at +Y, base on plane y=-1/3.
    const TETRA_V = [
      [0,           1,       0      ],   // 0 apex
      [0,          -1/3,     0.9428 ],   // 1 base front  (2√2/3)
      [-0.8165,    -1/3,    -0.4714 ],   // 2 base back-left  (√(2/3), √2/3)
      [ 0.8165,    -1/3,    -0.4714 ],   // 3 base back-right
    ];
    // 4 triangular faces (vertex indices, CCW from outside).
    const TETRA_F = [
      [0, 1, 3],  // front-right
      [0, 2, 1],  // front-left
      [0, 3, 2],  // back
      [1, 2, 3],  // bottom
    ];

    // Light direction (world space) — upper-right + slightly forward.
    const LIGHT = (() => {{
      const lx = 0.5, ly = 0.65, lz = 0.55;
      const len = Math.sqrt(lx*lx + ly*ly + lz*lz);
      return [lx/len, ly/len, lz/len];
    }})();

    function drawTetra(p, alpha) {{
      const cx = p.x;
      const cy = p.y;
      const r = p.r;
      // Per-particle independent rotation around Y then X. Each particle's
      // angular velocity is its own — neighbouring tetrahedra tumble at
      // different rates and phases, no two look the same. (Same effect
      // as picking a random per-particle rotation axis but cheaper to
      // compute than full Rodrigues.)
      const cosY = Math.cos(p.rotY);
      const sinY = Math.sin(p.rotY);
      const cosX = Math.cos(p.rotX);
      const sinX = Math.sin(p.rotX);

      const rotated = [null, null, null, null];
      const projected = [null, null, null, null];
      for (let i = 0; i < 4; i++) {{
        const [vx, vy, vz] = TETRA_V[i];
        // Y-axis rotation
        const x1 = vx * cosY + vz * sinY;
        const y1 = vy;
        const z1 = -vx * sinY + vz * cosY;
        // X-axis rotation (per-particle)
        const x2 = x1;
        const y2 = y1 * cosX - z1 * sinX;
        const z2 = y1 * sinX + z1 * cosX;
        rotated[i] = [x2, y2, z2];
        projected[i] = [cx + x2 * r, cy - y2 * r];
      }}

      // Compute per-face normal + centroid z + shade. Cull back faces.
      const drawList = [];
      for (const [ia, ib, ic] of TETRA_F) {{
        const a = rotated[ia], b = rotated[ib], c = rotated[ic];
        const ux = b[0]-a[0], uy = b[1]-a[1], uz = b[2]-a[2];
        const vx = c[0]-a[0], vy = c[1]-a[1], vz = c[2]-a[2];
        let nx = uy*vz - uz*vy;
        let ny = uz*vx - ux*vz;
        let nz = ux*vy - uy*vx;
        const nlen = Math.sqrt(nx*nx + ny*ny + nz*nz) || 1;
        nx /= nlen; ny /= nlen; nz /= nlen;
        // Back-face cull: viewer looks down +Z, faces with normal.z <= 0
        // point away — skip them.
        if (nz <= 0.02) continue;
        const dot = nx*LIGHT[0] + ny*LIGHT[1] + nz*LIGHT[2];
        const shade = 0.30 + 0.70 * Math.max(0, dot);
        const cz = (a[2] + b[2] + c[2]) / 3;
        drawList.push({{ ia, ib, ic, shade, cz }});
      }}
      // Painter's algorithm — back to front (smaller z first).
      drawList.sort((a, b) => a.cz - b.cz);

      for (const f of drawList) {{
        const v0 = projected[f.ia];
        const v1 = projected[f.ib];
        const v2 = projected[f.ic];
        ctx.fillStyle = p.color + (alpha * f.shade) + ')';
        ctx.beginPath();
        ctx.moveTo(v0[0], v0[1]);
        ctx.lineTo(v1[0], v1[1]);
        ctx.lineTo(v2[0], v2[1]);
        ctx.closePath();
        ctx.fill();
        // Crisp edge stroke for silhouette / face boundary clarity.
        ctx.strokeStyle = p.color + (alpha * 0.85) + ')';
        ctx.lineWidth = Math.max(0.5, r * 0.06);
        ctx.stroke();
      }}
    }}

    function ensureSized() {{
      // Use the iframe's own viewport, NOT the canvas element's
      // offsetWidth — when the canvas only has CSS `inset:0` (no
      // explicit width/height), some browsers leave it at the
      // intrinsic 300×150, which is what was trapping particles in
      // the top-left corner. window.innerWidth/Height inside an
      // iframe = its content area, always correct after layout.
      const w = window.innerWidth || document.documentElement.clientWidth;
      const h = window.innerHeight || document.documentElement.clientHeight;
      if (w < 30 || h < 30) return false;

      // Update both the drawing buffer (canvas.width/height) AND the
      // display size (CSS px) so they're always in sync. Setting only
      // canvas.width without style would scale-stretch the buffer.
      if (w !== bg.width) {{
        bg.width = w;
        bg.style.width = w + 'px';
      }}
      if (h !== bg.height) {{
        bg.height = h;
        bg.style.height = h + 'px';
      }}

      if (w !== lastW || h !== lastH) {{
        const N = Math.min(80, Math.max(40, Math.round(w * h / 22000)));
        if (lastW === 0) {{
          // First real size — create the full set scattered everywhere.
          parts = [];
          for (let i = 0; i < N; i++) parts.push(makeParticle(w, h));
        }} else {{
          // Resize — top up / trim, redistribute out-of-bounds particles.
          while (parts.length < N) parts.push(makeParticle(w, h));
          if (parts.length > N) parts.length = N;
          for (const p of parts) {{
            if (p.x > w) p.x = Math.random() * w;
            if (p.y > h) p.y = Math.random() * h;
          }}
        }}
        lastW = w;
        lastH = h;
      }}
      return true;
    }}

    let t = 0;
    let drawOrder = [];   // particles sorted by z (far → near) for painter's algorithm
    function rebuildDrawOrder() {{
      drawOrder = parts.slice().sort((a, b) => a.z - b.z);
    }}

    function tick() {{
      if (!ensureSized()) {{
        requestAnimationFrame(tick);
        return;
      }}
      if (drawOrder.length !== parts.length) rebuildDrawOrder();
      t += 0.012;
      ctx.clearRect(0, 0, bg.width, bg.height);

      // Pass 1: faint connections between nearby particles, drawn FIRST
      // so even foreground particles render on top. Connection alpha is
      // gated by both endpoints' depth — far↔far links almost invisible,
      // near↔near links most visible. Reinforces the parallax depth cue.
      ctx.lineWidth = 0.8;
      for (let i = 0; i < parts.length; i++) {{
        for (let j = i + 1; j < parts.length; j++) {{
          const dx = parts[i].x - parts[j].x;
          const dy = parts[i].y - parts[j].y;
          const d2 = dx * dx + dy * dy;
          if (d2 < 14000) {{
            const depthBoost = (parts[i].z + parts[j].z) * 0.5;  // 0..1
            const a = (1 - d2 / 14000) * 0.12 * (0.4 + depthBoost);
            ctx.strokeStyle = 'rgba(122, 162, 247, ' + a + ')';
            ctx.beginPath();
            ctx.moveTo(parts[i].x, parts[i].y);
            ctx.lineTo(parts[j].x, parts[j].y);
            ctx.stroke();
          }}
        }}
      }}

      // Pass 2: tetrahedra drawn far → near (painter's algorithm) with
      // z-driven glow on the closer ones for a "floating in space" feel.
      for (const p of drawOrder) {{
        p.x += p.vx;
        p.y += p.vy;
        p.rotX += p.rotXSpeed;     // independent per-particle tumble
        p.rotY += p.rotYSpeed;
        if (p.x < -p.r * 2) p.x = bg.width + p.r * 2;
        if (p.x > bg.width + p.r * 2) p.x = -p.r * 2;
        if (p.y < -p.r * 2) p.y = bg.height + p.r * 2;
        if (p.y > bg.height + p.r * 2) p.y = -p.r * 2;
        const a = Math.max(0, p.a0 + 0.10 * Math.sin(t + p.phase));
        if (p.z > 0.4) {{
          ctx.shadowBlur = 6 + p.z * 14;
          ctx.shadowColor = p.color + (a * 0.9) + ')';
        }} else {{
          ctx.shadowBlur = 0;
        }}
        drawTetra(p, a);
      }}
      ctx.shadowBlur = 0;
      requestAnimationFrame(tick);
    }}
    tick();
  }})();

  const ELEMENTS = {payload};

  // Task 3 (2026-05-11): backend precomputes positions via networkx
  // spring_layout (see src/dashboard/layout_cache.py) and ships them in
  // each node's `position`. Frontend uses `preset` so cytoscape just
  // plants nodes — no in-browser force-directed iteration, no main-
  // thread freeze even at 5K+ nodes.
  const cy = cytoscape({{
    container: document.getElementById('cy'),
    elements: ELEMENTS,
    layout: {{ name: 'preset', fit: true, padding: 30 }},
    minZoom: 0.05,
    maxZoom: 100,
    wheelSensitivity: 0.25,
    style: [
      {{
        selector: 'node',
        style: {{
          'background-color': 'data(color)',
          'label': 'data(label)',
          'color': '#c0caf5',
          'font-weight': 500,
          // Label outline halo + vertical offset are also counter-scaled
          // — otherwise at zoom > 1 the 2-model-unit outline turns into a
          // 6+ px black ring around every character (adjacent characters'
          // outlines merge into a solid black blob) and labels drift far
          // away from their nodes. cs_outline / cs_tmy track scale just
          // like cs_w / cs_font.
          'text-outline-width': 'data(cs_outline)',
          'text-outline-color': '#16171f',
          'text-margin-y': 'data(cs_tmy)',
          'border-color': 'rgba(255,255,255,0.18)',
          'overlay-padding': 4,
          'transition-property': 'border-color',
          'transition-duration': '120ms',
          // All size dimensions are data()-driven from per-element fields
          // (cs_w / cs_b / cs_font etc., precomputed by the JS zoom
          // handler). cytoscape re-evaluates `data()` references on
          // every render, INCLUDING when :selected / [?highlight] rules
          // turn on — so size is always the current zoom-counter-scaled
          // value regardless of how selection state toggled. There is no
          // inline `n.style({{...}})` anywhere; that approach got wiped
          // by cytoscape on selection state changes and was the root
          // cause of the "selected node stays big" / "border disappears"
          // bugs.
          'width':        'data(cs_w)',
          'height':       'data(cs_w)',
          'font-size':    'data(cs_font)',
          'border-width': 'data(cs_b)',
        }},
      }},
      {{
        // Task 3 B: hidden by the viewport degree-threshold logic.
        // `display: none` removes them from layout / picking entirely.
        selector: 'node.threshold-hidden, edge.threshold-hidden',
        style: {{ 'display': 'none' }},
      }},
      {{
        // Audit-focus highlight (set at Python render time via
        // `highlight_node_ids`). Always-on for the session, never toggles
        // — safe to bump size. Red ring distinguishes it from the
        // yellow user-selection ring below.
        selector: 'node[?highlight]',
        style: {{
          'border-color': '#f7768e',
          'width':        'data(cs_w_hl)',
          'height':       'data(cs_w_hl)',
          'border-width': 'data(cs_b_hl)',
        }},
      }},
      {{
        selector: 'node[?is_ghost]',
        style: {{
          'opacity': 0.45,
          'border-style': 'dashed',
        }},
      }},
      {{
        // User selection — yellow ring + thicker border. NO size change
        // (size stays driven by the base node rule's data()). Selection
        // visual is purely additive: cytoscape toggles border-color and
        // border-width on/off cleanly, no events to miss, no inline-
        // style ghosts.
        selector: 'node:selected',
        style: {{
          'border-color': '#ffd166',
          'border-width': 'data(cs_b_sel)',
        }},
      }},
      {{
        selector: 'edge',
        style: {{
          'curve-style': 'bezier',
          'line-color': 'rgba(160, 173, 209, 0.22)',
          'target-arrow-shape': 'triangle',
          'target-arrow-color': 'rgba(160, 173, 209, 0.34)',
          'opacity': 0.85,
          'width':       'data(cs_ew)',
          'arrow-scale': 'data(cs_arrow)',
        }},
      }},
      {{
        selector: 'edge[?highlight]',
        style: {{
          'line-color': '#f7768e',
          'target-arrow-color': '#f7768e',
          'opacity': 1.0,
          'width': 'data(cs_ew_hl)',
        }},
      }},
      {{
        // Edge selection — yellow line + slightly thicker. Like nodes,
        // size/width is data()-driven so it tracks zoom correctly even
        // through selection state changes.
        selector: 'edge:selected',
        style: {{
          'line-color': '#ffd166',
          'target-arrow-color': '#ffd166',
          'width': 'data(cs_ew_sel)',
        }},
      }},
    ],
  }});

  const card = document.getElementById('card');

  function escapeHtml(s) {{
    const div = document.createElement('div');
    div.textContent = s == null ? '' : String(s);
    return div.innerHTML;
  }}

  function renderCard(d) {{
    const type = escapeHtml(d.type || 'node');
    const label = escapeHtml(d.label || '(unlabeled)');
    const summary = escapeHtml(d.summary || '(no summary)');
    const source = escapeHtml(d.source_title || '');
    const sourceQuote = escapeHtml(d.source_quote || '');
    const detail = escapeHtml(d.detail || '');
    const ghostNote = d.is_ghost
      ? `<div style="margin-top:6px;color:#f7768e;font-size:11px;">Retired (merged into ${{escapeHtml(d.merged_into || '')}}).</div>`
      : '';
    const sourceLine = source
      ? `<div class="card-source"><span class="card-source-label">Source:</span> ${{source}}</div>`
      : '';
    const moreFields = [
      ['Type',        type],
      ['Version',     escapeHtml(d.version)],
      ['Weight',      escapeHtml(d.weight)],
      ['Bucket',      escapeHtml(d.bucket)],
    ];
    const more = `
      <div class="more-toggle" onclick="(function(el){{
            const m = el.parentNode.querySelector('.more-content');
            m.classList.toggle('visible');
            el.textContent = m.classList.contains('visible')
              ? 'less ▴' : 'more ▾';
        }})(this)">more ▾</div>
      <div class="more-content">
        ${{moreFields.map(([k, v]) =>
          `<div class="field"><span class="field-name">${{k}}</span><br><span class="field-value">${{v}}</span></div>`
        ).join('')}}
        ${{sourceQuote ? `<div class="field"><span class="field-name">Source quote</span><div class="quote">${{sourceQuote}}</div></div>` : ''}}
        ${{detail ? `<div class="field"><span class="field-name">Detail (truncated)</span><div class="field-value">${{detail}}</div></div>` : ''}}
      </div>`;
    return `
      <span class="card-type">${{type}}</span>
      <div class="card-label">${{label}}</div>
      <div class="card-summary">${{summary}}</div>
      ${{sourceLine}}
      ${{ghostNote}}
      ${{more}}
    `;
  }}

  function positionCard(node) {{
    const pos = node.renderedPosition();
    const cy_rect = document.getElementById('cy').getBoundingClientRect();
    const card_rect = card.getBoundingClientRect();
    let x = pos.x + 22;
    let y = pos.y + 12;
    // Keep card inside the canvas
    if (x + card_rect.width > cy_rect.width - 8) {{
      x = pos.x - card_rect.width - 22;
    }}
    if (y + card_rect.height > cy_rect.height - 8) {{
      y = cy_rect.height - card_rect.height - 8;
    }}
    if (x < 8) x = 8;
    if (y < 8) y = 8;
    card.style.left = x + 'px';
    card.style.top  = y + 'px';
  }}

  cy.on('tap', 'node', evt => {{
    const node = evt.target;
    card.classList.remove('visible');
    setTimeout(() => {{
      card.innerHTML = renderCard(node.data());
      positionCard(node);
      card.classList.add('visible');
    }}, 80);
  }});
  cy.on('tap', evt => {{
    if (evt.target === cy) {{
      card.classList.remove('visible');
      cy.elements().unselect();
    }}
  }});
  cy.on('pan zoom', () => {{
    const sel = cy.$(':selected');
    if (sel.nonempty() && card.classList.contains('visible')) {{
      positionCard(sel[0]);
    }}
  }});

  // ---- Task 3 B: viewport-aware degree threshold ----
  // Keep at most VIEWPORT_NODE_CAP nodes visible. The threshold T is the
  // smallest degree such that #(nodes in viewport with degree >= T) <= cap.
  // When zoomed out (many nodes in viewport), T rises; when zoomed into a
  // sparse region, T falls and more low-degree nodes appear.
  const VIEWPORT_NODE_CAP = 120;
  const THRESHOLD_OFF_BELOW_TOTAL = 150;  // small subgraphs skip the cap entirely
  const ALL_NODES = cy.nodes();
  const TOTAL_ELIGIBLE = ALL_NODES.length;
  const THRESHOLD_ENABLED = TOTAL_ELIGIBLE > THRESHOLD_OFF_BELOW_TOTAL;

  function pickThreshold() {{
    const ext = cy.extent();  // model-coord bbox of current viewport
    const counts = new Map();
    ALL_NODES.forEach(n => {{
      const p = n.position();
      if (p.x < ext.x1 || p.x > ext.x2 || p.y < ext.y1 || p.y > ext.y2) return;
      const d = n.data('degree') || 0;
      counts.set(d, (counts.get(d) || 0) + 1);
    }});
    const degs = [...counts.keys()].sort((a, b) => b - a);
    if (degs.length === 0) return Infinity;
    let cumulative = 0;
    let chosen = degs[0];  // safe default: show top tier even if it exceeds cap
    for (const d of degs) {{
      const next = cumulative + counts.get(d);
      if (next > VIEWPORT_NODE_CAP) break;
      cumulative = next;
      chosen = d;
    }}
    return chosen;
  }}

  let lastThreshold = -1;
  function applyThreshold() {{
    if (!THRESHOLD_ENABLED) return;
    const T = pickThreshold();
    if (T === lastThreshold) return;
    lastThreshold = T;
    cy.startBatch();
    ALL_NODES.forEach(n => {{
      const d = n.data('degree') || 0;
      if (d >= T) {{
        n.removeClass('threshold-hidden');
      }} else {{
        n.addClass('threshold-hidden');
      }}
    }});
    cy.edges().forEach(e => {{
      if (e.source().hasClass('threshold-hidden') || e.target().hasClass('threshold-hidden')) {{
        e.addClass('threshold-hidden');
      }} else {{
        e.removeClass('threshold-hidden');
      }}
    }});
    cy.endBatch();
  }}

  // Debounce pan/zoom so we don't recompute every frame while the user drags.
  let _thrTimer = null;
  cy.on('pan zoom', () => {{
    if (_thrTimer) clearTimeout(_thrTimer);
    _thrTimer = setTimeout(applyThreshold, 150);
  }});

  // ---- Task 3 (cont.): zoom-counter-scale via precomputed data fields ----
  // Past zoom ≈ 1.0 we want on-screen size of nodes/edges to stay
  // constant (otherwise they grow with zoom and re-overlap). For each
  // visual dimension the stylesheet binds `data(cs_<dim>)` — a per-
  // element pre-scaled value. On every zoom event we recompute these
  // (BASE × counter-scale) and write them via `n.data({{...}})`.
  // cytoscape's data-event mechanism then re-runs every applicable
  // style rule, including :selected and [?highlight], so size always
  // tracks the current zoom regardless of selection toggling.
  const FREEZE_ZOOM = 1.0;
  const NODE_BASE_W = 22, NODE_HL_W = 28;
  const NODE_BASE_FONT = 11;
  const NODE_BASE_B = 1.5, NODE_SEL_B = 3.5, NODE_HL_B = 3;
  const TEXT_OUTLINE_BASE = 2;
  const TEXT_MARGIN_Y_BASE = -6;
  const EDGE_BASE_W = 1.1, EDGE_SEL_W = 2.2, EDGE_HL_W = 2.0;
  const ARROW_BASE = 0.7;

  function _currentScale() {{
    const z = cy.zoom();
    return z > FREEZE_ZOOM ? (FREEZE_ZOOM / z) : 1.0;
  }}

  let _lastScale = -1;
  function applyAllScale() {{
    const s = _currentScale();
    if (Math.abs(s - _lastScale) < 0.01) return;
    _lastScale = s;
    const nodeFields = {{
      cs_w: NODE_BASE_W * s,
      cs_w_hl: NODE_HL_W * s,
      cs_font: NODE_BASE_FONT * s,
      cs_b: NODE_BASE_B * s,
      cs_b_sel: NODE_SEL_B * s,
      cs_b_hl: NODE_HL_B * s,
      cs_outline: TEXT_OUTLINE_BASE * s,
      cs_tmy: TEXT_MARGIN_Y_BASE * s,
    }};
    const edgeFields = {{
      cs_ew: EDGE_BASE_W * s,
      cs_ew_sel: EDGE_SEL_W * s,
      cs_ew_hl: EDGE_HL_W * s,
      cs_arrow: ARROW_BASE * s,
    }};
    cy.startBatch();
    cy.nodes().forEach(n => n.data(nodeFields));
    cy.edges().forEach(e => e.data(edgeFields));
    cy.endBatch();
  }}
  let _scalePending = false;
  function scheduleAllScale() {{
    if (_scalePending) return;
    _scalePending = true;
    requestAnimationFrame(() => {{
      _scalePending = false;
      applyAllScale();
    }});
  }}
  cy.on('zoom', scheduleAllScale);

  // Initial pass — runs AFTER all the const/let/function declarations
  // above so applyAllScale has access to its closures (FREEZE_ZOOM,
  // _lastScale, etc.). cy.ready guarantees the preset layout + fit are
  // settled so cy.zoom() returns the fitted zoom and cy.extent() returns
  // the correct viewport bbox. Both functions are accurate here.
  // Earlier we had this block BEFORE the declarations; cytoscape fires
  // ready synchronously (since the graph is already ready by the time
  // we register), which hit a TDZ on the const/let names and threw a
  // silent ReferenceError — that prevented the zoom listener below from
  // registering and broke counter-scale entirely.
  cy.ready(() => {{
    applyAllScale();
    applyThreshold();
  }});
</script>
</body>
</html>
"""
