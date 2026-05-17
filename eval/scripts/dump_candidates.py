"""Dump deterministic, ranked candidate sets for hand-curated testset writing.

Reads the production graph + text_units, produces:
  - single_hop_edges    : high-quality semantic edges with evidence + chunks
  - multi_hop_bridges   : 2-hop paths where chunks are disjoint + different docs
                          + no direct A-C edge
  - hub_nodes           : top-N nodes by total degree, with typed neighbor lists

Output: eval/testset/candidates.json (read by Claude in-conversation when
writing eval/testset/gold.jsonl).

This is data sampling only — no LLM, no judgement. Determinism via fixed
seed so re-running yields the same candidate pool.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.graph.storage import GraphStorage
from src.graph.text_units import TextUnitStore

SEED = 42
SINGLE_HOP_PER_TYPE = 4
SINGLE_HOP_MAX_TOTAL = 60
MULTI_HOP_MAX = 40
HUB_TOP_N = 15
CHUNK_SAMPLE_N = 50

# Edge types we treat as "noisy / not from ontology" — skip for single-hop sampling.
# Anything not in this allow-list is filtered. This is the conservative ontology
# core (intentionally excludes RELATED_TO since RELATED_TO edges are mostly
# domain-mismatch downgrades per §16.19).
ALLOWED_EDGE_TYPES = {
    "AFFILIATED_WITH", "HEADQUARTERED_IN", "PARTNERED_WITH", "MEMBERSHIP_OF",
    "IN_INDUSTRY", "PART_OF", "LOCATED_IN", "STUDIED_LOCATION", "OPERATES_IN",
    "PRODUCES", "AUTHORED_WITH", "AUTHORED_BY", "OWNED_BY", "CO_FOUNDED",
    "CEO_OF", "AUDITED_BY", "AUDITS", "PUBLISHED_BY", "PRODUCED_BY",
    "FUNDED_BY", "FUNDED", "FOUNDED", "PRODUCED_IN", "EXPORTS", "INCLUDES",
    "IMPORTED_FROM", "CONTAINS", "PUBLISHED", "REGULATES",
}


def _good_single_hop(edge, src_node, tgt_node, text_units) -> bool:
    if edge.type not in ALLOWED_EDGE_TYPES:
        return False
    if not edge.evidence_quote or len(edge.evidence_quote) < 20:
        return False
    if not edge.text_unit_ids:
        return False
    if src_node is None or tgt_node is None:
        return False
    if src_node.merged_into is not None or tgt_node.merged_into is not None:
        return False
    # All chunk_ids must resolve in text_units.
    for cid in edge.text_unit_ids:
        if cid not in text_units:
            return False
    return True


def _chunk_doc(chunk_id: str, text_units) -> str | None:
    cd = text_units.get(chunk_id)
    return cd.get("raw_doc_id") if cd else None


def _normalize_doc(rid: str) -> str:
    """Strip super-chunk suffix so '#pages_X_Y' variants count as one doc."""
    if not rid:
        return rid
    idx = rid.find("#pages_")
    return rid[:idx] if idx >= 0 else rid


def _build_adjacency(storage):
    """Build {node_id: [(neighbor_id, edge)]} for both directions, all edges."""
    adj = defaultdict(list)
    for e in storage.edges():
        adj[e.source_id].append((e.target_id, e, "out"))
        adj[e.target_id].append((e.source_id, e, "in"))
    return adj


def _node_payload(n) -> dict:
    return {
        "id": n.id,
        "label": n.label,
        "type": n.type,
        "summary": n.summary,
    }


def _edge_payload(e, text_units) -> dict:
    docs = sorted({
        _normalize_doc(_chunk_doc(c, text_units) or "")
        for c in e.text_unit_ids
        if _chunk_doc(c, text_units)
    })
    pages_set = set()
    for c in e.text_unit_ids:
        cd = text_units.get(c)
        if cd and cd.get("page_num") is not None:
            pages_set.add(cd["page_num"])
    pages = sorted(pages_set)
    return {
        "id": e.id,
        "type": e.type,
        "evidence_quote": e.evidence_quote,
        "text_unit_ids": list(e.text_unit_ids),
        "raw_doc_ids": docs,
        "pages": pages,
        "weight": e.weight,
    }


def main():
    random.seed(SEED)

    storage = GraphStorage(ROOT / "data" / "production" / "graph.pkl")
    storage.load()
    text_units = TextUnitStore(ROOT / "data" / "production" / "text_units.json")
    text_units.load()

    nodes_by_id = {n.id: n for n in storage.nodes()}

    # ---- 1. single_hop_edges (stratified by edge_type) ----
    by_type: dict[str, list] = defaultdict(list)
    for e in storage.edges():
        src = nodes_by_id.get(e.source_id)
        tgt = nodes_by_id.get(e.target_id)
        if not _good_single_hop(e, src, tgt, text_units):
            continue
        by_type[e.type].append(e)

    single_hop = []
    for et in sorted(by_type):
        pool = by_type[et]
        random.shuffle(pool)
        for e in pool[:SINGLE_HOP_PER_TYPE]:
            single_hop.append({
                "edge": _edge_payload(e, text_units),
                "src": _node_payload(nodes_by_id[e.source_id]),
                "tgt": _node_payload(nodes_by_id[e.target_id]),
            })
    random.shuffle(single_hop)
    single_hop = single_hop[:SINGLE_HOP_MAX_TOTAL]

    # ---- 2. multi_hop_bridges ----
    # Brute force: for each "good" edge e1=(A,B), for each "good" edge e2=(B,C)
    # incident to B, check filters. Cap candidates per A to 1 to keep diversity.
    adj = _build_adjacency(storage)
    direct_pairs = set()
    for e in storage.edges():
        s, t = sorted([e.source_id, e.target_id])
        direct_pairs.add((s, t))

    good_edges = []
    for et_pool in by_type.values():
        good_edges.extend(et_pool)
    good_edge_ids = {e.id for e in good_edges}

    # Diversity caps:
    #   - each B (hub) appears in at most MAX_PER_HUB bridges
    #   - each e2 edge id used at most once
    #   - prefer hubs with smaller degree (stronger "true bridge" feel)
    MAX_PER_HUB = 3
    bridges = []
    seen_A = set()
    used_e2 = set()
    bridges_per_hub: dict[str, int] = defaultdict(int)
    random.shuffle(good_edges)
    # Order hubs by ascending degree so non-mega-hub bridges land first.
    # Re-sort good_edges so e1's that touch lower-degree hubs come first.
    def _hub_score(e):
        return min(
            len(adj.get(e.source_id, [])),
            len(adj.get(e.target_id, [])),
        )
    good_edges.sort(key=_hub_score)
    for e1 in good_edges:
        if len(bridges) >= MULTI_HOP_MAX * 4:
            break
        for endpoint_a, endpoint_b in [(e1.source_id, e1.target_id),
                                       (e1.target_id, e1.source_id)]:
            if endpoint_a in seen_A:
                continue
            if bridges_per_hub[endpoint_b] >= MAX_PER_HUB:
                continue
            for nbr_id, e2, _dir in adj.get(endpoint_b, []):
                if e2.id == e1.id or e2.id not in good_edge_ids:
                    continue
                if e2.id in used_e2:
                    continue
                if nbr_id == endpoint_a:
                    continue
                # Reject A==C by label too (alias-like duplicates).
                if nodes_by_id[nbr_id].label.strip().lower() == nodes_by_id[endpoint_a].label.strip().lower():
                    continue
                pair = tuple(sorted([endpoint_a, nbr_id]))
                if pair in direct_pairs:
                    continue
                c1 = set(e1.text_unit_ids)
                c2 = set(e2.text_unit_ids)
                if c1 & c2:
                    continue
                d1 = {
                    _normalize_doc(_chunk_doc(c, text_units) or "")
                    for c in e1.text_unit_ids
                }
                d2 = {
                    _normalize_doc(_chunk_doc(c, text_units) or "")
                    for c in e2.text_unit_ids
                }
                if d1 & d2:
                    continue
                A = nodes_by_id[endpoint_a]
                B = nodes_by_id[endpoint_b]
                C = nodes_by_id[nbr_id]
                bridges.append({
                    "A": _node_payload(A),
                    "B": _node_payload(B),
                    "C": _node_payload(C),
                    "edge1": _edge_payload(e1, text_units),
                    "edge2": _edge_payload(e2, text_units),
                    "edge1_direction": "A->B" if e1.source_id == endpoint_a else "B->A",
                    "edge2_direction": "B->C" if e2.source_id == endpoint_b else "C->B",
                    "B_degree": len(adj.get(endpoint_b, [])),
                })
                seen_A.add(endpoint_a)
                used_e2.add(e2.id)
                bridges_per_hub[endpoint_b] += 1
                break
            if endpoint_a in seen_A:
                break

    random.shuffle(bridges)
    bridges = bridges[:MULTI_HOP_MAX]

    # ---- 3. hub_nodes ----
    degree = {n.id: 0 for n in storage.nodes() if n.merged_into is None}
    for e in storage.edges():
        if e.source_id in degree:
            degree[e.source_id] += 1
        if e.target_id in degree:
            degree[e.target_id] += 1
    top_hubs = sorted(degree.items(), key=lambda x: -x[1])[:HUB_TOP_N]

    hub_payload = []
    for nid, deg in top_hubs:
        n = nodes_by_id[nid]
        nbrs_by_type: dict[str, list] = defaultdict(list)
        seen = set()
        for nb_id, e, _dir in adj.get(nid, []):
            if nb_id in seen:
                continue
            seen.add(nb_id)
            nb = nodes_by_id.get(nb_id)
            if nb is None or nb.merged_into is not None:
                continue
            nbrs_by_type[nb.type].append({
                "label": nb.label,
                "type": nb.type,
                "edge_type": e.type,
            })
        hub_payload.append({
            "node": _node_payload(n),
            "degree": deg,
            "neighbors_by_type": dict(nbrs_by_type),
        })

    # ---- 4. chunk samples (for me to understand corpus) ----
    all_cids = sorted(text_units.keys())
    random.shuffle(all_cids)
    chunk_samples = []
    for cid in all_cids[:CHUNK_SAMPLE_N]:
        cd = text_units.get(cid)
        if cd is None:
            continue
        chunk_samples.append({
            "id": cid,
            "raw_doc_id": cd.get("raw_doc_id", ""),
            "page_num": cd.get("page_num"),
            "text_preview": (cd.get("text", "") or "")[:300],
        })

    out = {
        "stats": {
            "total_active_nodes": sum(1 for n in storage.nodes() if n.merged_into is None),
            "total_edges": sum(1 for _ in storage.edges()),
            "total_text_units": len(text_units),
            "single_hop_count": len(single_hop),
            "multi_hop_count": len(bridges),
            "hub_count": len(hub_payload),
            "single_hop_types_covered": sorted(by_type.keys()),
        },
        "single_hop_edges": single_hop,
        "multi_hop_bridges": bridges,
        "hub_nodes_for_aggregation": hub_payload,
        "chunk_samples": chunk_samples,
    }

    out_path = ROOT / "eval" / "testset" / "candidates.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"wrote {out_path}")
    print(json.dumps(out["stats"], indent=2))


if __name__ == "__main__":
    main()
