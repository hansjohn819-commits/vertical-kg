"""Semantic retrieval (Phase 5 + §16.10).

Two layers:
1. `encode_node` / `encode_query` — single source of truth for the embedding
   text format. Anyone who writes vectors into the FAISS store or queries
   it MUST go through these, otherwise the index gets inconsistent text
   conventions.
2. `top_k` — Q&A retrieval that hits the per-instance FAISS index instead
   of re-encoding the whole graph each call (was O(N) per query).
3. `display_title` — convert internal raw_doc_id to a human-readable title
   for citations (drops .pdf / #pages_X_Y, replaces -/_ with spaces).

Embedding model: sentence-transformers/all-MiniLM-L6-v2, dim=384.
"""

from __future__ import annotations

import re

import numpy as np

from src.graph.models import Node
from src.graph.storage import GraphStorage
from src.graph.vector_store import VectorStore

# Multilingual variant — dim still 384 so FAISS index structure is
# unchanged (any rebuild against existing index file is required, but
# vector_store.py / EMBEDDING_DIM stay put). Swapped from
# `all-MiniLM-L6-v2` (English-only) on 2026-05-06: external chat with
# Chinese queries was returning cos<0.27 random matches because the
# old model treated Chinese tokens as noise. This variant is trained
# on parallel paraphrase data across 50+ languages and lifts cross-
# lingual cosine into the 0.6-0.8 range that matches monolingual
# English performance. M1 still emits English summaries; the
# improvement comes at query-encoding time when users type non-English.
_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2 fixed dim
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _node_text(node: Node) -> str:
    """Canonical embedding text — must match what the FAISS index was built
    against. Format chosen to carry the strongest signals: type prefix
    helps disambiguate same-label different-type entities.
    """
    return f"{node.type}: {node.label} — {node.summary}"


def encode_node(node: Node) -> np.ndarray:
    """Encode a node into a normalized float32 vector for FAISS IP=cosine."""
    v = _get_model().encode(
        _node_text(node), convert_to_numpy=True, normalize_embeddings=True
    )
    return v.astype(np.float32)


def encode_query(text: str) -> np.ndarray:
    """Encode a free-form query string into a normalized float32 vector."""
    v = _get_model().encode(
        text, convert_to_numpy=True, normalize_embeddings=True
    )
    return v.astype(np.float32)


def top_k(
    vector_store: VectorStore,
    storage: GraphStorage,
    question: str,
    k: int = 5,
) -> list[Node]:
    """FAISS top-k by cosine. Filters out ghosts and missing-from-storage hits."""
    if vector_store.size == 0:
        return []
    qv = encode_query(question)
    hits = vector_store.query(qv, k)
    nodes: list[Node] = []
    for nid, _score in hits:
        n = storage.get_node(nid)
        if n is None or n.merged_into is not None:
            continue
        nodes.append(n)
    return nodes


def with_neighbors(storage: GraphStorage, seeds: list[Node]) -> list[Node]:
    """Expand seed nodes with 1-hop neighbors, dedup, preserve seed order first."""
    seen: dict[str, Node] = {n.id: n for n in seeds}
    for n in seeds:
        for nb in storage.neighbors(n.id):
            if nb.merged_into is None:
                seen.setdefault(nb.id, nb)
    return list(seen.values())


_PAGES_SUFFIX_RE = re.compile(r"#pages_\d+_\d+$")


def display_title(raw_doc_id: str) -> str:
    """Convert an internal raw_doc_id (filename) into a human-readable title.

    Drops the file extension, drops any `#pages_X_Y` chunk fragment, and
    converts `-` / `_` to spaces. Used by external (and optionally internal)
    answer composers so users see "FAO SOFIA 2024" instead of
    "FAO SOFIA 2024.pdf#pages_1_110".

    Idempotent on already-clean strings; preserves whitespace and
    capitalization within tokens (we don't title-case to avoid
    "U.S." → "U.s." and similar mangling).
    """
    if not raw_doc_id:
        return ""
    s = _PAGES_SUFFIX_RE.sub("", raw_doc_id)
    # Strip a single trailing extension if present.
    if "." in s:
        head, tail = s.rsplit(".", 1)
        if 1 <= len(tail) <= 5 and tail.isalnum():
            s = head
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s
