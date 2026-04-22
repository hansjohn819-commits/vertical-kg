"""4a: prune. Guide §5.6.

v1 candidate policy (toy-friendly, deterministic — no LLM):
- Edges below PRUNE_WEIGHT_FLOOR after 4c → mark suspicious + downweight.
- Edges already suspicious: increment suspicious_pass_count; if >=
  PRUNE_DELETE_AFTER, delete and log.
- Node-level prune: orphaned ghosts from merges are cleaned by 4b's
  _cleanup_prior_ghosts next pass.

Per user decision: convergence is mechanical — if a round deletes nothing
new, the loop stops.
"""

from src.graph.instance import GraphInstance

from .pass_log import log_event
from .state import PRUNE_DELETE_AFTER, PRUNE_DOWNWEIGHT, PassState

PRUNE_WEIGHT_FLOOR = 0.1  # edges at/under this are candidates after 4c decay


def prune_step(state: PassState, *, instance: GraphInstance) -> dict:
    storage = instance.storage
    pass_id = state.get("pass_id", "unknown")
    iter_idx = int(state.get("prune_iter", 0))

    deleted = 0
    newly_marked = 0

    for e in list(storage.edges()):
        if e.weight <= PRUNE_WEIGHT_FLOOR:
            if e.suspicious:
                e.suspicious_pass_count += 1
                if e.suspicious_pass_count >= PRUNE_DELETE_AFTER:
                    storage.remove_edge_by_id(e.id)
                    deleted += 1
                    log_event({
                        "kind": "prune",
                        "pass_id": pass_id,
                        "summary": f"deleted edge {e.type} (weight={e.weight:.3f})",
                        "edge_id": e.id,
                        "src": e.source_id,
                        "tgt": e.target_id,
                    })
            else:
                e.suspicious = True
                e.suspicious_pass_count = 1
                e.weight *= PRUNE_DOWNWEIGHT
                newly_marked += 1

    stats = dict(state.get("stats") or {})
    stats["prune_total"] = int(stats.get("prune_total", 0)) + deleted
    stats[f"prune_round_{iter_idx}"] = {"deleted": deleted, "newly_marked": newly_marked}

    changed = deleted > 0

    log_event({
        "kind": "prune_round_done",
        "pass_id": pass_id,
        "summary": f"round {iter_idx + 1}: deleted {deleted}, marked {newly_marked}",
    })

    return {
        "prune_iter": iter_idx + 1,
        "prune_changed": changed,
        "stats": stats,
    }


def prune_should_continue(state: PassState) -> str:
    return "prune" if state.get("prune_changed") else "link"
