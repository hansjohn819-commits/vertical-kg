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
) -> dict:
    """Build {nodes: [...], edges: [...]} for cytoscape.

    `focus_node_ids` — if provided, only emit these nodes (+ optional 1-hop
                       neighbors when `add_neighbor_hop`). Used by past-event
                       focused-subgraph view.
    `highlight_node_ids` / `highlight_edge_ids` — visually flagged
                       (CSS class `highlight`) but rendered the same way.

    Active nodes only (skip ghosts where merged_into is set), unless the
    ghost is explicitly in focus_node_ids (so a past merge event can show
    the now-retired originals).
    """
    focus = focus_node_ids
    if focus and add_neighbor_hop:
        expanded: set[str] = set(focus)
        for nid in list(focus):
            for nb in storage.neighbors(nid):
                expanded.add(nb.id)
        focus = expanded

    nodes_out: list[dict] = []
    seen_node_ids: set[str] = set()
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
        nodes_out.append({
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
            },
        })
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
            },
        })

    return {"nodes": nodes_out, "edges": edges_out}


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
      - fcose layout (force-directed, smooth)
      - Pan / zoom built-in
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
<script src="https://unpkg.com/layout-base@2.0.1/layout-base.js"></script>
<script src="https://unpkg.com/cose-base@2.2.0/cose-base.js"></script>
<script src="https://unpkg.com/cytoscape-fcose@2.2.0/cytoscape-fcose.js"></script>
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

  const cy = cytoscape({{
    container: document.getElementById('cy'),
    elements: ELEMENTS,
    layout: {{
      name: 'fcose',
      animate: true,
      randomize: true,
      idealEdgeLength: 95,
      nodeRepulsion: 8500,
      gravity: 0.18,
      gravityCompound: 1.5,
      numIter: 2500,
      tile: true,
      packComponents: true,
    }},
    minZoom: 0.2,
    maxZoom: 2.2,
    wheelSensitivity: 0.25,
    style: [
      {{
        selector: 'node',
        style: {{
          'background-color': 'data(color)',
          'label': 'data(label)',
          'color': '#c0caf5',
          'font-size': 11,
          'font-weight': 500,
          'text-outline-width': 2,
          'text-outline-color': '#16171f',
          'text-margin-y': -6,
          'width': 22, 'height': 22,
          'border-width': 1.5,
          'border-color': 'rgba(255,255,255,0.18)',
          'overlay-padding': 4,
          'transition-property': 'background-color, border-color, width, height',
          'transition-duration': '120ms',
        }},
      }},
      {{
        selector: 'node[?highlight]',
        style: {{
          'border-color': '#ffd166',
          'border-width': 3,
          'width': 28, 'height': 28,
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
        selector: 'node:selected',
        style: {{
          'border-color': '#7aa2f7',
          'border-width': 3.5,
          'width': 28, 'height': 28,
        }},
      }},
      {{
        selector: 'edge',
        style: {{
          'curve-style': 'bezier',
          'width': 1.1,
          'line-color': 'rgba(160, 173, 209, 0.22)',
          'target-arrow-shape': 'triangle',
          'target-arrow-color': 'rgba(160, 173, 209, 0.34)',
          'arrow-scale': 0.7,
          'opacity': 0.85,
        }},
      }},
      {{
        selector: 'edge[?highlight]',
        style: {{
          'line-color': '#ffd166',
          'target-arrow-color': '#ffd166',
          'width': 2.0,
          'opacity': 1.0,
        }},
      }},
      {{
        selector: 'edge:selected',
        style: {{
          'line-color': '#7aa2f7',
          'target-arrow-color': '#7aa2f7',
          'width': 2.2,
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
</script>
</body>
</html>
"""
