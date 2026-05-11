"""One-off graph maintenance: re-run ontology validation over every edge.

Mirrors what M1 ingest and the §16.17 round-2 link_form fix do at edge-
creation time, but applied retroactively to a graph that was built
before those validations were in place. Useful after:

  * Adding a new alias to ``ontology.md`` — sweep existing edges to
    rewrite stale typo types.
  * Extending or tightening a relation's ``domain:`` line — sweep to
    downgrade out-of-domain edges to ``RELATED_TO`` (with audit logs).
  * Wiring a previously-bypassing code path (e.g., M4d link_form) into
    the validator — sweep to bring its earlier output in line.

Per edge, the operation is:

  edge.type = _classify_edge_type(
      edge.type, src.type, src.label, tgt.type, tgt.label,
      edge.evidence_quote, ontology_rels, domain_map, log_event,
      raw_doc_id="(retro)", run_id="(normalize_edges)",
      pass_label="retro_cleanup",
      alias_map=aliases,
  )

If the classifier rewrites the type, the edge's stored type is updated.
Side-effect log_events (alias_resolved / ontology_proposal) flow into
``log.md`` so the audit trail captures the cleanup.

Usage:
    python scripts/normalize_edges.py [--dry-run]

Idempotent: a second run is a no-op once the graph is consistent with
the current ontology. After running normalize_edges you typically want
to follow up with ``scripts/dedupe_edges.py`` because alias rewrites
can collapse previously-distinct (src, tgt, type) tuples.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from src.graph.instance import GraphInstance
from src.modules.m1_ingest import (
    _classify_edge_type,
    parse_aliases,
    parse_ontology,
)
from src.modules.m4_sleep_pass.pass_log import log_event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report rewrites without modifying the graph.")
    parser.add_argument("--instance", default="production")
    args = parser.parse_args(argv)

    instance_dir = _WORKSPACE_ROOT / "data" / args.instance
    ontology_path = _WORKSPACE_ROOT / "ontology.md"
    if not instance_dir.exists():
        print(f"error: {instance_dir} does not exist", file=sys.stderr)
        return 1

    gi = GraphInstance(
        name=args.instance, storage_path=instance_dir, ontology_path=ontology_path,
    )
    ontology_text = ontology_path.read_text(encoding="utf-8")
    _ents, rels, dom = parse_ontology(ontology_text)
    aliases = parse_aliases(ontology_text)

    edges_total = sum(1 for _ in gi.storage.edges())
    print(f"Loaded {args.instance}: {edges_total} edges, "
          f"{sum(1 for _ in gi.storage.nodes())} nodes")
    print(f"Ontology: {len(rels)} relation types, "
          f"{len(aliases)} alias{'es' if len(aliases) != 1 else ''}, "
          f"{len(dom)} domain entries")

    rewrites: list[tuple[str, str, str, str]] = []  # (old, new, src_label, tgt_label)
    sink_events = [] if args.dry_run else None
    sink = sink_events.append if sink_events is not None else log_event

    for edge in list(gi.storage.edges()):
        src = gi.storage.get_node(edge.source_id)
        tgt = gi.storage.get_node(edge.target_id)
        if src is None or tgt is None:
            continue
        new_type = _classify_edge_type(
            edge.type, src.type, src.label, tgt.type, tgt.label,
            edge.evidence_quote, rels, dom, sink,
            raw_doc_id="(retro)", run_id="(normalize_edges)",
            page_num=None, pass_label="retro_cleanup",
            alias_map=aliases,
        )
        if new_type != edge.type:
            rewrites.append((edge.type, new_type, src.label, tgt.label))
            if not args.dry_run:
                edge.type = new_type

    print(f"\nRewrites: {len(rewrites)}")
    if rewrites:
        by_pair = Counter((old, new) for old, new, _, _ in rewrites)
        print("By (old → new):")
        for (old, new), c in by_pair.most_common(20):
            print(f"  {c:3d}× {old}  →  {new}")
        print("\nFirst 5 rewrites with endpoints:")
        for old, new, sl, tl in rewrites[:5]:
            print(f"  {old}  →  {new}    ({sl!r} -> {tl!r})")

    if args.dry_run:
        print("\n[dry-run] No changes saved.")
        if sink_events:
            counts = Counter(e.get("kind") for e in sink_events)
            print(f"would-emit log events: {dict(counts)}")
    elif rewrites:
        gi.save()
        print("\nSaved.")
    else:
        print("\nNothing to do — graph already aligned with ontology.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
