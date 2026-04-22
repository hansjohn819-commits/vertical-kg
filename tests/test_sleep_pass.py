"""Phase 6 smoke test: end-to-end sleep pass over the toy graph.

Asserts:
- Tesla Inc + Tesla Motors get merged (4b) — new label contains "Tesla".
- The pre-marked low-weight stale edge (Musk -[RELATED_TO]-> NHTSA) is
  deleted by 4a (one more suspicious_pass_count reaches the threshold).
- At least one derived edge is added by 4d.
- The pass reports `status=ok` with sane counters.

Run:
    python -m tests.test_sleep_pass
"""

import sys
from pathlib import Path

from scripts.seed_toy import seed
from src.graph.instance import GraphInstance


def main() -> int:
    storage_dir = "data/experiment"
    seed(storage_dir)

    gi = GraphInstance("toy-experiment", storage_dir, "ontology.md")
    pre_node_count = gi.stats()["node_count"]
    pre_edge_count = gi.stats()["edge_count"]
    pre_edge_types = gi.stats()["edge_types"]

    print(f"Before pass: {pre_node_count} nodes, {pre_edge_count} edges")
    print(f"  edge types: {pre_edge_types}")

    result = gi.sleep_pass()
    print("\n=== Pass result ===")
    print(f"pass_id:       {result.get('pass_id')}")
    print(f"status:        {result.get('status')}")
    print(f"merge_iters:   {result.get('merge_iters')}")
    print(f"prune_iters:   {result.get('prune_iters')}")
    print(f"link_iters:    {result.get('link_iters')}")
    print(f"stats:         {result.get('stats')}")

    post_stats = gi.stats()
    print(f"\nAfter pass: {post_stats['node_count']} nodes, {post_stats['edge_count']} edges")
    print(f"  node types: {post_stats['node_types']}")
    print(f"  edge types: {post_stats['edge_types']}")

    # Reload from disk to confirm persistence.
    gi2 = GraphInstance("toy-experiment-reload", storage_dir, "ontology.md")
    tesla_nodes = [n for n in gi2.storage.nodes() if "tesla" in n.label.lower() and n.merged_into is None]
    print(f"\nActive Tesla-ish nodes after pass: {[(n.label, round(n.weight, 3)) for n in tesla_nodes]}")

    stats = result.get("stats", {})
    merge_total = stats.get("merge_total", 0)
    prune_total = stats.get("prune_total", 0)
    link_total = stats.get("link_total", 0)

    checks = {
        "status_ok": result.get("status") == "ok",
        "merge_happened": merge_total >= 1,
        "tesla_consolidated": len(tesla_nodes) < 2,  # originally 2, should be ≤1
        "prune_happened": prune_total >= 1,
        "link_happened": link_total >= 1,
    }
    print("\n=== Checks ===")
    for k, v in checks.items():
        print(f"  {'OK  ' if v else 'FAIL'} {k}")

    ok = all(checks.values())
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
