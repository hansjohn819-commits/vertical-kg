"""One-off graph maintenance: collapse duplicate edges in production.

Same logic the M1 ingest pipeline runs at end-of-document, but applied
*globally* across every edge in the loaded GraphInstance. Use it to
clean up legacy graphs that pre-date the §16.17 dedup wiring (or any
graph where you suspect duplicate-edge buildup, e.g. after multiple
re-ingests of the same source document).

Operation per duplicate group (same `source_id, target_id, type`):
  * keep the lowest-id edge as survivor
  * union all `text_unit_ids` into the survivor
  * sum all `weight`s into the survivor
  * pick the longest `evidence_quote` across the group as the survivor's
  * concatenate retraction logs

Saves on success, prints a one-line summary. Idempotent: a second run
on the same instance is a no-op.

Usage:
    python scripts/dedupe_edges.py [--dry-run]

`--dry-run` lists the duplicate groups it would merge and exits without
writing anything to disk.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from src.graph.instance import GraphInstance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report duplicate groups without modifying the graph.",
    )
    parser.add_argument(
        "--instance", default="production",
        help="Instance name (default: production)",
    )
    args = parser.parse_args(argv)

    instance_dir = _WORKSPACE_ROOT / "data" / args.instance
    ontology_path = _WORKSPACE_ROOT / "ontology.md"
    if not instance_dir.exists():
        print(f"error: {instance_dir} does not exist", file=sys.stderr)
        return 1

    gi = GraphInstance(
        name=args.instance,
        storage_path=instance_dir,
        ontology_path=ontology_path,
    )
    edges_before = sum(1 for _ in gi.storage.edges())
    print(f"Loaded {args.instance}: {edges_before} edges, "
          f"{sum(1 for _ in gi.storage.nodes())} nodes")

    if args.dry_run:
        groups: dict[tuple[str, str, str], list] = defaultdict(list)
        for e in gi.storage.edges():
            groups[(e.source_id, e.target_id, e.type)].append(e)
        dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
        type_counter = Counter(k[2] for k in dup_groups)
        total_excess = sum(len(v) - 1 for v in dup_groups.values())
        print(f"\n[dry-run] {len(dup_groups)} duplicate groups, "
              f"{total_excess} excess edges would be removed")
        if dup_groups:
            print(f"\nBy edge type:")
            for t, c in type_counter.most_common():
                print(f"  {c:4d}  {t}")
            print(f"\nFirst 10 groups (src_id → tgt_id [type]: count):")
            for (src, tgt, t), edges in list(dup_groups.items())[:10]:
                src_node = gi.storage.get_node(src)
                tgt_node = gi.storage.get_node(tgt)
                src_label = src_node.label if src_node else "?"
                tgt_label = tgt_node.label if tgt_node else "?"
                print(f"  {len(edges):3d}× {src_label!r} -[{t}]-> {tgt_label!r}")
        return 0

    stats = gi.storage.dedupe_edges()
    edges_after = sum(1 for _ in gi.storage.edges())
    print(f"\nDedup result: merged {stats['groups_merged']} groups, "
          f"removed {stats['edges_removed']} duplicate edges "
          f"({edges_before} → {edges_after})")

    if stats["edges_removed"] > 0:
        gi.save()
        print("Saved.")
    else:
        print("Nothing to do — graph is already clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
