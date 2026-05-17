"""Build a BM25 index over production graph nodes for external_v2.

Each node becomes a single document with text:
    "<type>: <label> — <summary>"

(Same canonical text format as src.graph.retrieval._node_text, which is what
the FAISS index already uses.) Persisted to eval/data/node_bm25/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.baseline.bm25_store import BaselineBM25Store
from src.graph.storage import GraphStorage


def main():
    storage = GraphStorage(ROOT / "data" / "production" / "graph.pkl")
    storage.load()

    ids: list[str] = []
    texts: list[str] = []
    for n in storage.nodes():
        if n.merged_into is not None:
            continue
        text = f"{n.type}: {n.label} — {n.summary}"
        ids.append(n.id)
        texts.append(text)

    out_dir = ROOT / "eval" / "data" / "node_bm25"
    out_dir.mkdir(parents=True, exist_ok=True)
    bm25 = BaselineBM25Store()
    bm25.fit(ids, texts)
    bm25.save(out_dir)
    print(f"wrote node BM25 over {len(ids)} active nodes -> {out_dir}")


if __name__ == "__main__":
    main()
