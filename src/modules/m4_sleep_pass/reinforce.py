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

    # Apply bumps + global decay.
    for e in list(storage.edges()):
        tr = traverse_hits.get(e.id, 0)
        ci = cite_hits.get(e.id, 0)
        e.weight = (e.weight + tr * DELTA_TRAVERSE + ci * DELTA_CITE) * DECAY

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
