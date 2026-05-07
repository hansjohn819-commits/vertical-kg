"""FAISS-backed vector store keyed by string node_ids.

Wraps `IndexFlatIP` (cosine via normalized inner product) inside an
`IndexIDMap2` so `remove_ids` actually works — the bare flat index has no
remove path. Internal int64 IDs are auto-incremented; the str↔int mapping
is persisted alongside the index so node ids survive process restarts.

Used by:
- `src/graph/retrieval.py:top_k` — Q&A retrieval (replaces full-graph encode)
- `src/modules/m4_sleep_pass/merge.py:_candidate_pairs` — M4b candidates
- M1 ingest hook — add new nodes
- M4b merge hook — add fused / remove originals
- M4b ghost cleanup — idempotent remove tombstone

Maintenance contract (guide §16.10.3, audited 2026-05-06):
- type/label/summary on a node are immutable post-creation; no update path
- M4a prune touches only edges, not nodes — vector store untouched
- ghosts (merged_into != None) must NOT be in the index — removal happens
  at merge-time, not at the §5.5 one-pass-delayed cleanup
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, index_path: Path | str, ids_path: Path | str, dim: int):
        self.index_path = Path(index_path)
        self.ids_path = Path(ids_path)
        self.dim = dim
        self._index: faiss.IndexIDMap2 = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._id_to_int: dict[str, int] = {}
        self._int_to_id: dict[int, str] = {}
        self._next_int_id: int = 0

    # --- Mutations -------------------------------------------------------

    def add(self, node_id: str, vec: np.ndarray) -> None:
        """Add a node vector. Idempotent: re-adding the same id is a no-op.

        `vec` must already be L2-normalized so inner-product == cosine.
        """
        if node_id in self._id_to_int:
            return
        v = np.ascontiguousarray(np.asarray(vec, dtype=np.float32).reshape(1, -1))
        if v.shape[1] != self.dim:
            raise ValueError(f"vec dim {v.shape[1]} != index dim {self.dim}")
        int_id = self._next_int_id
        self._next_int_id += 1
        self._index.add_with_ids(v, np.array([int_id], dtype=np.int64))
        self._id_to_int[node_id] = int_id
        self._int_to_id[int_id] = node_id

    def remove(self, node_id: str) -> bool:
        """Remove by node_id. Idempotent: missing id returns False, no raise."""
        int_id = self._id_to_int.pop(node_id, None)
        if int_id is None:
            return False
        self._index.remove_ids(np.array([int_id], dtype=np.int64))
        del self._int_to_id[int_id]
        return True

    # --- Queries ---------------------------------------------------------

    def query(self, vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return up to k (node_id, cosine_score) pairs, descending by score."""
        if self._index.ntotal == 0 or k <= 0:
            return []
        v = np.ascontiguousarray(np.asarray(vec, dtype=np.float32).reshape(1, -1))
        k_eff = min(k, self._index.ntotal)
        scores, ids = self._index.search(v, k_eff)
        out: list[tuple[str, float]] = []
        for score, int_id in zip(scores[0], ids[0]):
            if int_id < 0:
                continue
            node_id = self._int_to_id.get(int(int_id))
            if node_id is not None:
                out.append((node_id, float(score)))
        return out

    def has(self, node_id: str) -> bool:
        return node_id in self._id_to_int

    @property
    def size(self) -> int:
        return self._index.ntotal

    # --- Persistence -----------------------------------------------------

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        meta = {
            "dim": self.dim,
            "id_to_int": self._id_to_int,
            "next_int_id": self._next_int_id,
        }
        self.ids_path.write_text(json.dumps(meta), encoding="utf-8")

    def load(self) -> bool:
        """Return True iff a coherent index was loaded.

        False means caller should call `rebuild_from(...)`. We never raise
        on a corrupt index file — boot-time auto-rebuild (§16.10.5) is the
        recovery path, not a hard error.
        """
        if not self.index_path.exists() or not self.ids_path.exists():
            return False
        try:
            idx = faiss.read_index(str(self.index_path))
            meta = json.loads(self.ids_path.read_text(encoding="utf-8"))
            if int(meta.get("dim", 0)) != self.dim:
                return False
            self._index = idx
            self._id_to_int = {k: int(v) for k, v in meta["id_to_int"].items()}
            self._int_to_id = {v: k for k, v in self._id_to_int.items()}
            self._next_int_id = int(meta.get("next_int_id", len(self._id_to_int)))
            return True
        except Exception:
            return False

    def rebuild_from(self, pairs: list[tuple[str, np.ndarray]]) -> None:
        """Wipe and rebuild from (node_id, vec) pairs.

        Caller is responsible for filtering ghosts (merged_into is None)
        and for computing already-normalized vectors.
        """
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
        self._id_to_int.clear()
        self._int_to_id.clear()
        self._next_int_id = 0
        for nid, vec in pairs:
            self.add(nid, vec)
