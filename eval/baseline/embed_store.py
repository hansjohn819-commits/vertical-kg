"""Independent FAISS dense store for the baseline.

Same embedding model as the main project (paraphrase-multilingual-MiniLM-L12-v2)
because that's what's installed and validated locally; the comparison stays
about the *retrieval architecture* (graph + neighbor expansion vs flat RAG),
not about embedding model quality. The model is loaded fresh inside this
module so there's no shared state with src.graph.retrieval.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    vecs = _get_model().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return vecs.astype(np.float32)


class BaselineDenseStore:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []  # row -> chunk_id

    def add(self, chunk_ids: list[str], vecs: np.ndarray) -> None:
        self.index.add(vecs)
        self.ids.extend(chunk_ids)

    def query(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        qvec = qvec.reshape(1, -1).astype(np.float32)
        D, I = self.index.search(qvec, min(k, self.index.ntotal))
        out = []
        for score, row in zip(D[0], I[0]):
            if row < 0:
                continue
            out.append((self.ids[row], float(score)))
        return out

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / "dense.faiss"))
        (dir_path / "dense_ids.json").write_text(
            json.dumps(self.ids), encoding="utf-8"
        )

    def load(self, dir_path: Path) -> None:
        self.index = faiss.read_index(str(dir_path / "dense.faiss"))
        self.ids = json.loads((dir_path / "dense_ids.json").read_text(encoding="utf-8"))


# ---- chunk metadata persistence ----

def save_chunks(chunks_meta: list[dict], dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    with (dir_path / "chunks.pkl").open("wb") as f:
        pickle.dump(chunks_meta, f)


def load_chunks(dir_path: Path) -> list[dict]:
    with (dir_path / "chunks.pkl").open("rb") as f:
        return pickle.load(f)
