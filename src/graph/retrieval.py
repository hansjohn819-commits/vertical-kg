"""Semantic retrieval for graph_query (Phase 5).

v1: sentence-transformers + numpy cosine. Lazy-load model so tool-calling
tests (which only exercise routing, not retrieval) don't pay embedding cost.
"""

import numpy as np

from src.graph.models import Node
from src.graph.storage import GraphStorage

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def top_k(storage: GraphStorage, question: str, k: int = 5) -> list[Node]:
    nodes = list(storage.nodes())
    if not nodes:
        return []
    model = _get_model()
    texts = [f"{n.label}: {n.summary}" for n in nodes]
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    q = model.encode(question, convert_to_numpy=True, normalize_embeddings=True)
    scores = embs @ q
    idx = np.argsort(-scores)[: min(k, len(nodes))]
    return [nodes[i] for i in idx]


def with_neighbors(storage: GraphStorage, seeds: list[Node]) -> list[Node]:
    """Expand seed nodes with 1-hop neighbors, dedup, preserve seed order first."""
    seen: dict[str, Node] = {n.id: n for n in seeds}
    for n in seeds:
        for nb in storage.neighbors(n.id):
            seen.setdefault(nb.id, nb)
    return list(seen.values())
