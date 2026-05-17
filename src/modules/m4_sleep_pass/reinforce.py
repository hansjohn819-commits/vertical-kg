"""4c: reinforce / decay. Guide §5.4.

Single-shot: read traversal.jsonl since last pass, bump edges that were
traversed or cited, apply global decay, recompute node weights, then clear
the traversal log.

Node weight policy (v1): average of incident edge weights (guide §5.4 v1).
"""

from src.graph.instance import GraphInstance
from src.graph.traversal_log import clear as clear_log
from src.graph.traversal_log import read_all

from .pass_log import log_event
from .state import DECAY, DELTA_CITE, DELTA_TRAVERSE, PassState


def reinforce_step(state: PassState, *, instance: GraphInstance) -> dict:
    storage = instance.storage
    records = read_all(storage)

    # Build per-edge bump counts from traversal log.
    traverse_hits: dict[str, int] = {}
    cite_hits: dict[str, int] = {}
    for rec in records:
        for eid in rec.get("touched_edge_ids", []):
            traverse_hits[eid] = traverse_hits.get(eid, 0) + 1
        # cite bump applies to edges incident to any seed node
        seeds = set(rec.get("seed_node_ids", []))
        for eid in rec.get("touched_edge_ids", []):
            edge = storage.get_edge(eid)
            if edge is None:
                continue
            if edge.source_id in seeds or edge.target_id in seeds:
                cite_hits[eid] = cite_hits.get(eid, 0) + 1

    # Apply bumps + global decay. §16.8.1: only emit per-edge log lines for
    # edges that actually got a traverse/cite bump — every other edge just
    # took the global decay, which is uniform and not worth a line each.
    pass_id = state.get("pass_id")
    for e in list(storage.edges()):
        tr = traverse_hits.get(e.id, 0)
        ci = cite_hits.get(e.id, 0)
        old_weight = e.weight
        e.weight = (e.weight + tr * DELTA_TRAVERSE + ci * DELTA_CITE) * DECAY
        if tr or ci:
            src = storage.get_node(e.source_id)
            tgt = storage.get_node(e.target_id)
            log_event({
                "kind": "reinforce_edge",
                "pass_id": pass_id,
                "summary": (
                    f"{src.label if src else e.source_id} -[{e.type}]-> "
                    f"{tgt.label if tgt else e.target_id}: "
                    f"{old_weight:.4f} → {e.weight:.4f} (tr={tr}, ci={ci})"
                ),
                "edge_id": e.id,
                "edge_type": e.type,
                "source_id": e.source_id,
                "source_label": src.label if src else None,
                "target_id": e.target_id,
                "target_label": tgt.label if tgt else None,
                "old_weight": round(old_weight, 4),
                "new_weight": round(e.weight, 4),
                "traverse_hits": tr,
                "cite_hits": ci,
            })

    # Recompute node weights = avg of incident edge weights (fall back to 1.0).
    for n in list(storage.nodes()):
        incident = storage.incident_edges(n.id)
        if incident:
            n.weight = sum(e.weight for e in incident) / len(incident)
        # else: leave seeded weight alone

    # Clear log — next pass starts fresh.
    clear_log(storage)

    stats = dict(state.get("stats") or {})
    stats["reinforce_records_consumed"] = len(records)
    stats["reinforce_edges_traversed"] = len(traverse_hits)

    log_event({
        "kind": "reinforce",
        "pass_id": state.get("pass_id"),
        "summary": f"consumed {len(records)} Q&A records, touched {len(traverse_hits)} edges",
    })

    return {"stats": stats}
