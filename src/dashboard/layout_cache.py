"""Precomputed graph layout cache for the audit view (§16.6 task 3).

The audit view used to run `cytoscape-fcose` in the browser on every
page load. That works for ~200 nodes but freezes the page at 2000+
because force-directed layout cost grows roughly O(N log N) per
iteration × ~2500 iterations on the JS main thread.

This module moves layout to Python:
  * `compute_layout(storage)` — runs networkx spring_layout
    (Fruchterman-Reingold, scipy-accelerated) over active non-orphan
    nodes. Deterministic seed → stable positions across reloads.
  * `save_layout` / `load_layout` — JSON persistence per instance.
  * `ensure_layout(instance_dir, storage)` — returns cached positions
    if the cache covers the current node set exactly; otherwise
    recomputes and writes a fresh cache.

`GraphInstance.save()` calls `ensure_layout` so layout is refreshed
whenever the graph topology changes (M1 ingest, M4 sleep pass).
Page-load cost in the browser drops from "5-15s freeze" to "instant"
because cytoscape just plants nodes at the precomputed coordinates
via `layout: { name: 'preset' }`.

Coordinate convention: positions normalized to a square roughly
[-1000, 1000] × [-1000, 1000] in cytoscape model units. The exact
range doesn't matter because the viewport auto-fits on first render.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.graph.storage import GraphStorage

_LAYOUT_FILENAME = "layout.json"
_LAYOUT_SCHEMA_VERSION = 1
# Seed makes layout reproducible — same graph topology yields the same
# positions, so users build positional memory of the graph.
_LAYOUT_SEED = 42
# Coordinate scale for cytoscape preset layout. 1000 keeps small graphs
# zoomable and gives big graphs room to spread.
_LAYOUT_SCALE = 1000.0
# spring_layout iterations. 80 converges well for graphs up to ~5K
# nodes; bump if quality drops on larger graphs.
_LAYOUT_ITERATIONS = 80


def _eligible_node_ids(storage: GraphStorage) -> list[str]:
    """Nodes that should participate in layout — active (not ghost) and
    non-orphan (degree >= 1). Matches the filter in
    `graph_to_cytoscape`, so cached positions cover every node the
    frontend can render."""
    out: list[str] = []
    for n in storage.nodes():
        if n.merged_into is not None:
            continue
        if storage.degree(n.id) == 0:
            continue
        out.append(n.id)
    return out


def compute_layout(storage: GraphStorage) -> dict[str, tuple[float, float]]:
    """Run networkx spring_layout (Fruchterman-Reingold) over eligible
    nodes. Returns ``{node_id: (x, y)}`` in cytoscape model coordinates.

    The graph is built fresh each call from the current storage state.
    networkx handles parameter scaling internally via its ``k`` (optimal
    distance) default of ``1/sqrt(n)``, so no per-N tuning is needed.
    """
    import networkx as nx  # local — only the audit view depends on networkx

    eligible = set(_eligible_node_ids(storage))
    if not eligible:
        return {}

    G = nx.Graph()
    for nid in eligible:
        G.add_node(nid)
    for e in storage.edges():
        if e.source_id in eligible and e.target_id in eligible:
            G.add_edge(e.source_id, e.target_id)

    pos = nx.spring_layout(
        G,
        k=None,                      # auto-scale: 1/sqrt(n)
        iterations=_LAYOUT_ITERATIONS,
        seed=_LAYOUT_SEED,
        scale=_LAYOUT_SCALE,
    )
    return {nid: (float(xy[0]), float(xy[1])) for nid, xy in pos.items()}


def save_layout(
    instance_dir: Path, positions: dict[str, tuple[float, float]],
) -> None:
    """Write positions to ``<instance_dir>/layout.json``. Atomic via
    temp-file rename so a mid-write crash doesn't leave a half-file
    that triggers spurious recomputes."""
    instance_dir = Path(instance_dir)
    instance_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _LAYOUT_SCHEMA_VERSION,
        "positions": {nid: [x, y] for nid, (x, y) in positions.items()},
    }
    target = instance_dir / _LAYOUT_FILENAME
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)


def load_layout(instance_dir: Path) -> dict[str, tuple[float, float]] | None:
    """Read positions from ``<instance_dir>/layout.json``, or return
    ``None`` if missing/corrupt/wrong-schema."""
    path = Path(instance_dir) / _LAYOUT_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("schema_version") != _LAYOUT_SCHEMA_VERSION:
        return None
    raw = data.get("positions") or {}
    out: dict[str, tuple[float, float]] = {}
    for nid, xy in raw.items():
        if isinstance(xy, list) and len(xy) == 2:
            out[nid] = (float(xy[0]), float(xy[1]))
    return out


def ensure_layout(
    instance_dir: Path, storage: GraphStorage,
) -> dict[str, tuple[float, float]]:
    """Return cached positions if the cache covers exactly the current
    eligible node set; otherwise recompute, write, return fresh.

    "Covers exactly" = same set of node ids. Adding or removing any
    node invalidates. We don't bother with partial updates — for the
    expected size (1K-10K nodes) full recompute is cheap (1-5s) and
    stable layouts beat fast-but-jittery incremental ones.
    """
    eligible = set(_eligible_node_ids(storage))
    cached = load_layout(instance_dir)
    if cached is not None and set(cached.keys()) == eligible:
        return cached
    fresh = compute_layout(storage)
    save_layout(instance_dir, fresh)
    return fresh
