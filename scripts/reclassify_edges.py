"""One-off backfill: LLM-reclassify historical RELATED_TO edges.

Companion to the sleep-pass 4e sub-op (§16.19 c). 4e handles the
deterministic, zero-LLM cases: alias rewrites, domain downgrades, noise
sediment, and rescue of edges whose ``original_type`` was already
captured at write time. This script handles what 4e cannot — old
RELATED_TO edges from before ``original_type`` existed, where the
original LLM proposal is no longer recoverable from the edge itself
and only an LLM can re-propose a type from the evidence quote.

Default scope: ``e.type == "RELATED_TO" and e.original_type is None``.
That keeps us from re-running the LLM on edges already handled by 4e or
by a previous run of this script. Pass ``--all`` to sweep every
RELATED_TO regardless.

Usage:
    python scripts/reclassify_edges.py [--instance production] [--all]

Once ``--original_type`` is populated everywhere, this script's job is
basically done — a future ontology revision can rely on 4e alone to
rescue edges back to richer types.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from src.graph.instance import GraphInstance
from src.llm.local_client import LocalClient
from src.modules.m1_ingest import (
    _reclassify_related_to_edges,
    _split_ontology,
    parse_aliases,
    parse_ontology,
)
from src.modules.m4_sleep_pass.pass_log import log_event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default="production",
                        help="Graph instance dir under data/ (default: production).")
    parser.add_argument("--all", action="store_true",
                        help="Reclassify every RELATED_TO edge, including ones "
                             "whose original_type field is already populated.")
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
    _full, relation_types_block = _split_ontology(ontology_text)
    _ents, rels, dom = parse_ontology(ontology_text)
    aliases = parse_aliases(ontology_text)

    if args.all:
        scope = {e.id for e in gi.storage.edges() if e.type == "RELATED_TO"}
        scope_desc = "all RELATED_TO edges"
    else:
        scope = {
            e.id for e in gi.storage.edges()
            if e.type == "RELATED_TO" and e.original_type is None
        }
        scope_desc = "RELATED_TO edges with no original_type"

    total_related = sum(1 for e in gi.storage.edges() if e.type == "RELATED_TO")
    print(f"Loaded {args.instance}: {total_related} RELATED_TO edges total")
    print(f"Scope: {scope_desc} → {len(scope)} candidates")
    if not scope:
        print("Nothing to do.")
        return 0

    client = LocalClient()

    stats = _reclassify_related_to_edges(
        client, gi.storage, relation_types_block,
        rels, dom, log_event,
        raw_doc_id="(retro)", run_id="(reclassify_edges)",
        alias_map=aliases,
        edge_ids=scope,
    )

    print(f"\nReclassify stats: {stats}")

    gi.save()
    print("Saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
