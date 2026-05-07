"""Wipe and rebuild the FAISS vector index for an instance from storage.

Run when:
- First-time bootstrap on an existing graph that predates §16.10 (no index files)
- Suspected index corruption (ImportError in load, missing nodes in queries)
- After an embedding-model swap (text format change → all old vecs invalid)

Normal bootstrap is automatic — `GraphInstance.__init__` rebuilds when
load() fails or node counts mismatch. This script is the manual override
for edge cases.

Usage:
    python scripts/rebuild_faiss.py [--instance production|experiment]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.graph.instance import GraphInstance
from src.graph.retrieval import encode_node


def rebuild(instance_name: str) -> dict:
    storage_path = _ROOT / "data" / instance_name
    ontology_path = _ROOT / "ontology.md"
    if not storage_path.exists():
        raise SystemExit(f"data dir not found: {storage_path}")

    # Constructing the instance triggers consistency check + auto-rebuild
    # when the vector store is missing. We bypass that and force-rebuild
    # explicitly so users running this script see the work happening.
    gi = GraphInstance(name=instance_name, storage_path=storage_path,
                       ontology_path=ontology_path)
    active = [n for n in gi.storage.nodes() if n.merged_into is None]
    print(f"instance: {instance_name}")
    print(f"active nodes: {len(active)}")
    print(f"current index size: {gi.vector_store.size}")

    t0 = time.perf_counter()
    pairs = [(n.id, encode_node(n)) for n in active]
    encode_s = time.perf_counter() - t0
    print(f"encoded {len(pairs)} nodes in {encode_s:.1f}s")

    gi.vector_store.rebuild_from(pairs)
    gi.vector_store.save()
    print(f"rebuilt + saved. new index size: {gi.vector_store.size}")
    return {
        "instance": instance_name,
        "nodes": len(pairs),
        "encode_seconds": round(encode_s, 1),
        "index_size": gi.vector_store.size,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="production",
                   choices=["production", "experiment"])
    args = p.parse_args()
    rebuild(args.instance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
