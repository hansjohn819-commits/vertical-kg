"""BM25 lexical store for the baseline (rank_bm25).

Tokenization: simple lowercase + word-boundary split. No stemming / stopword
removal — keeps the baseline conventional and predictable; rank_bm25 BM25Okapi
already handles term saturation and length normalization.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOK_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOK_RE.findall((text or "").lower())


class BaselineBM25Store:
    def __init__(self):
        self.ids: list[str] = []
        self.bm25: BM25Okapi | None = None

    def fit(self, chunk_ids: list[str], texts: list[str]) -> None:
        self.ids = list(chunk_ids)
        tokens = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokens)

    def query(self, q: str, k: int) -> list[tuple[str, float]]:
        if self.bm25 is None:
            return []
        toks = tokenize(q)
        scores = self.bm25.get_scores(toks)
        # Top-k by score.
        order = scores.argsort()[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in order]

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        with (dir_path / "bm25.pkl").open("wb") as f:
            pickle.dump({"ids": self.ids, "bm25": self.bm25}, f)

    def load(self, dir_path: Path) -> None:
        with (dir_path / "bm25.pkl").open("rb") as f:
            d = pickle.load(f)
        self.ids = d["ids"]
        self.bm25 = d["bm25"]
