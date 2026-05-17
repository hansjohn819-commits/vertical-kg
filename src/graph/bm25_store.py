"""Node-level BM25 index (§16.20).

Companion to FAISS (vector_store.py). External-path retrieval fuses dense
top-K + BM25 top-K via RRF in top_k() — BM25 catches lexically distinctive
labels (`Last, First` bibliography style, rare proper nouns) that dense
embeddings miss because their tokenization treats abbreviations as noise.

Design notes:
  - rank_bm25's BM25Okapi has no incremental add/remove — the IDF table is
    fit once over the whole corpus. So we don't expose add/remove either;
    the only mutation is `fit_from(storage)` which re-fits the full active
    node set. Cost at 2.4k nodes is ~100ms — well within GraphInstance.save()
    budget (which is called once per ingest / once per sleep pass, not per
    add). This is dramatically simpler than FAISS's incremental maintenance
    and removes a whole class of "dirty tracking" bookkeeping.

  - Index text mirrors src.graph.retrieval._node_text exactly. The §16.10.3
    audit ("node label / summary are immutable after creation; M4b merge
    creates a new node + ghosts the old one, not in-place mutation") is
    reused as-is — BM25 inherits the same invariant.

  - Tokenization is intentionally simple (lowercase + r"\\w+", no stopwords,
    no stemming). Matches the BM25Okapi convention used elsewhere in
    industry and keeps the index conceptually transparent.

Persistence:
  - Single pickle file at data/<instance>/bm25.pkl.
  - Boot-time consistency check (see GraphInstance) rebuilds if file is
    missing/corrupt/size-mismatched against active node count.
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .retrieval import _node_text


_TOK_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOK_RE.findall((text or "").lower())


class BM25Store:
    """In-memory BM25Okapi index + pickle persistence.

    API mirrors vector_store.py where it makes sense, but DOES NOT expose
    add/remove — the underlying BM25Okapi requires a full refit on any
    change, so the only mutation is `fit_from(storage)`.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._ids: list[str] = []  # row index -> node_id
        self._bm25: BM25Okapi | None = None

    @property
    def size(self) -> int:
        return len(self._ids)

    # --- Build / refit ---

    def fit_from(self, storage) -> None:
        """Full refit from storage. Skips ghost nodes (merged_into is not None)
        — same exclusion as FAISS rebuild_from per §16.10.3 invariant.
        """
        pairs: list[tuple[str, str]] = []
        for n in storage.nodes():
            if n.merged_into is not None:
                continue
            pairs.append((n.id, _node_text(n)))

        self._ids = [p[0] for p in pairs]
        if not pairs:
            self._bm25 = None
            return
        tokenized = [_tokenize(p[1]) for p in pairs]
        self._bm25 = BM25Okapi(tokenized)

    # --- Query ---

    def query(self, question: str, k: int) -> list[tuple[str, float]]:
        """Return [(node_id, score)] descending by score, length ≤ k.
        Empty index → empty list (callers should treat as no hits)."""
        if self._bm25 is None or not self._ids:
            return []
        toks = _tokenize(question)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        order = scores.argsort()[::-1][:k]
        return [(self._ids[i], float(scores[i])) for i in order]

    # --- Persistence ---

    def save(self) -> None:
        """Atomic write: dump to .tmp then rename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump({"ids": self._ids, "bm25": self._bm25}, f)
        os.replace(tmp, self.path)

    def load(self) -> bool:
        """Load index from disk. Returns True iff a file existed and loaded
        successfully. Missing file → empty index (returns False); corrupt
        file → raises (caller decides whether to auto-rebuild). Boot
        consistency check is what handles the auto-rebuild path."""
        if not self.path.exists():
            self._ids = []
            self._bm25 = None
            return False
        with self.path.open("rb") as f:
            d = pickle.load(f)
        self._ids = list(d.get("ids", []))
        self._bm25 = d.get("bm25")
        return True
