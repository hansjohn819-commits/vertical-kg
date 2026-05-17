"""Public entry: run a full sleep pass synchronously.

Holds the instance's `sleep_pass_running` lock across the pass. The M2
agent checks the lock before handling requests and refuses chat while a
pass is in flight (Phase 6 sync semantics per user direction).
"""

from datetime import datetime, timezone
from uuid import uuid4

from src.graph.instance import GraphInstance

from .graph import build_pass_graph
from .pass_log import log_event


def run_sleep_pass(instance: GraphInstance) -> dict:
    if instance.sleep_pass_running:
        return {"status": "already_running", "pass_id": None}

    pass_id = f"pass-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
    instance.sleep_pass_running = True
    log_event({"kind": "pass_start", "pass_id": pass_id, "summary": f"instance={instance.name}"})
    try:
        app = build_pass_graph(instance)
        initial: dict = {
            "pass_id": pass_id,
            "merge_iter": 0,
            "prune_iter": 0,
            "link_iter": 0,
            "merge_done_vote": False,
            "prune_changed": False,
            "link_changed": False,
            "merged_new_ids": [],
            "merge_rejected_pairs": [],
            "seeded_for_link": [],
            "link_tried_pairs": [],
            "stats": {},
            "events": [],
        }
        # Bump recursion cap: merge(10) + prune(few) + link(few) + misc.
        final = app.invoke(initial, config={"recursion_limit": 100})
        instance.save()
        stats = final.get("stats", {})
        # §16.8.2: fixed-key counter dict so the agent can't ignore non-zero
        # fields. Round-level breakdowns stay under `stats` for tools that
        # want them (list_recent_merges, list_recent_prunings).
        merge_candidates_examined = sum(
            v.get("candidates", 0)
            for k, v in stats.items()
            if k.startswith("merge_round_") and isinstance(v, dict)
        )
        link_candidates_examined = sum(
            v.get("asked", 0)
            for k, v in stats.items()
            if k.startswith("link_round_") and isinstance(v, dict)
        )
        result = {
            "status": "ok",
            "pass_id": pass_id,
            "merges": int(stats.get("merge_total", 0)),
            "edges_pruned": int(stats.get("prune_total", 0)),
            "nodes_pruned": int(stats.get("nodes_pruned_total", 0)),
            "new_links": int(stats.get("link_total", 0)),
            "reinforced": int(stats.get("reinforce_edges_traversed", 0)),
            "merge_candidates_examined": int(merge_candidates_examined),
            "link_candidates_examined": int(link_candidates_examined),
            # Iteration counts + raw stats for callers who need detail.
            "merge_iters": final.get("merge_iter", 0),
            "prune_iters": final.get("prune_iter", 0),
            "link_iters": final.get("link_iter", 0),
            "stats": stats,
        }
        log_event({
            "kind": "pass_end",
            "pass_id": pass_id,
            "summary": (
                f"merges={result['merges']} edges_pruned={result['edges_pruned']} "
                f"nodes_pruned={result['nodes_pruned']} new_links={result['new_links']} "
                f"reinforced={result['reinforced']}"
            ),
        })
        return result
    finally:
        instance.sleep_pass_running = False


if __name__ == "__main__":
    gi = GraphInstance("toy-experiment", "data/experiment", "ontology.md")
    result = run_sleep_pass(gi)
    import json
    print(json.dumps(result, indent=2, default=str))
