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
            "seeded_for_link": [],
            "link_tried_pairs": [],
            "stats": {},
            "events": [],
        }
        # Bump recursion cap: merge(10) + prune(few) + link(few) + misc.
        final = app.invoke(initial, config={"recursion_limit": 100})
        instance.save()
        log_event({
            "kind": "pass_end",
            "pass_id": pass_id,
            "summary": f"merge={final['stats'].get('merge_total', 0)} "
                       f"prune={final['stats'].get('prune_total', 0)} "
                       f"link={final['stats'].get('link_total', 0)}",
        })
        return {
            "status": "ok",
            "pass_id": pass_id,
            "stats": final.get("stats", {}),
            "merge_iters": final.get("merge_iter", 0),
            "prune_iters": final.get("prune_iter", 0),
            "link_iters": final.get("link_iter", 0),
        }
    finally:
        instance.sleep_pass_running = False


if __name__ == "__main__":
    gi = GraphInstance("toy-experiment", "data/experiment", "ontology.md")
    result = run_sleep_pass(gi)
    import json
    print(json.dumps(result, indent=2, default=str))
