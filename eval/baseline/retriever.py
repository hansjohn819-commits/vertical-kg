"""Hybrid dense + BM25 retriever with Reciprocal Rank Fusion.

RRF formula:
    score(d) = sum over each retriever of 1 / (RRF_K + rank_in_retriever(d))

RRF_K = 60 is the conventional value (Cormack 2009).
"""

from __future__ import annotations

from .bm25_store import BaselineBM25Store
from .embed_store import BaselineDenseStore, embed_texts

RRF_K = 60
DENSE_K = 20
BM25_K = 20
FINAL_K = 10


def _rrf_combine(dense_hits: list[tuple[str, float]],
                 bm25_hits: list[tuple[str, float]],
                 k_final: int = FINAL_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, (cid, _) in enumerate(dense_hits):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, (cid, _) in enumerate(bm25_hits):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:k_final]


class HybridRetriever:
    def __init__(self,
                 dense: BaselineDenseStore,
                 bm25: BaselineBM25Store,
                 chunks_by_id: dict[str, dict]):
        self.dense = dense
        self.bm25 = bm25
        self.chunks_by_id = chunks_by_id

    def retrieve(self, question: str,
                 dense_k: int = DENSE_K,
                 bm25_k: int = BM25_K,
                 final_k: int = FINAL_K) -> list[dict]:
        qvec = embed_texts([question])[0]
        dense_hits = self.dense.query(qvec, dense_k)
        bm25_hits = self.bm25.query(question, bm25_k)
        fused = _rrf_combine(dense_hits, bm25_hits, final_k)
        out = []
        for cid, score in fused:
            cd = self.chunks_by_id.get(cid)
            if cd is None:
                continue
            out.append({**cd, "rrf_score": score})
        return out
