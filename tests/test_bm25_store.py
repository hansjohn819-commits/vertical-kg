"""Tests for the node-level BM25 store (§16.20).

Covers:
- fit_from(storage) excludes ghost nodes (merged_into != None)
- query(...) returns descending-score [(id, score)] lists, length ≤ k
- query on empty index returns empty list (no exception)
- save() + load() round-trip preserves ids and BM25 scores
- token-level scoring catches "Last, First" labels that dense embeddings miss
  (this is the whole reason BM25 exists in the pipeline)
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from src.graph.bm25_store import BM25Store
from src.graph.models import DirectProv, Node
from src.graph.storage import GraphStorage


def _mk_node(label: str, type_: str = "Person", summary: str = "",
             merged_into: str | None = None) -> Node:
    return Node(
        id=str(uuid4()),
        type=type_,
        label=label,
        summary=summary or f"{label} is a node",
        provenance=DirectProv(raw_doc_id="t.pdf", extraction_run_id="r0"),
        merged_into=merged_into,
    )


def _mk_storage(nodes: list[Node]) -> GraphStorage:
    with tempfile.TemporaryDirectory() as d:
        gs = GraphStorage(Path(d) / "graph.pkl")
    # In-memory only — no save/load — for unit testing.
    gs = GraphStorage("/dev/null/nope")  # path doesn't matter for in-memory ops
    for n in nodes:
        gs.add_node(n)
    return gs


def test_fit_from_excludes_ghosts():
    nodes = [
        _mk_node("Alice"),
        _mk_node("Bob"),
        _mk_node("OldAlice", merged_into="x123"),  # ghost
    ]
    gs = _mk_storage(nodes)

    bm = BM25Store("/dev/null/bm25.pkl")
    bm.fit_from(gs)

    assert bm.size == 2  # only the 2 active nodes
    active_ids = {n.id for n in nodes if n.merged_into is None}
    assert set(bm._ids) == active_ids


def test_query_returns_descending_scores():
    nodes = [
        _mk_node("Fujita, R.", summary="Marine biologist"),
        _mk_node("Stekoll, M.", summary="Kelp researcher"),
        _mk_node("Yarish, C.", summary="Seaweed expert collaborator"),
        _mk_node("Random Unrelated Person", summary="works on poultry"),
    ]
    gs = _mk_storage(nodes)
    bm = BM25Store("/dev/null/bm25.pkl")
    bm.fit_from(gs)

    hits = bm.query("Fujita kelp", k=4)
    assert len(hits) <= 4
    if len(hits) >= 2:
        # Scores must be sorted descending
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True)


def test_query_empty_index_returns_empty():
    bm = BM25Store("/dev/null/bm25.pkl")
    # Never fit
    assert bm.query("anything", k=5) == []


def test_query_empty_question_returns_empty():
    nodes = [_mk_node("Alice"), _mk_node("Bob")]
    gs = _mk_storage(nodes)
    bm = BM25Store("/dev/null/bm25.pkl")
    bm.fit_from(gs)
    assert bm.query("", k=5) == []


def test_save_load_round_trip(tmp_path):
    nodes = [_mk_node(f"Person{i}") for i in range(5)]
    gs = _mk_storage(nodes)

    p = tmp_path / "bm25.pkl"
    bm1 = BM25Store(p)
    bm1.fit_from(gs)
    bm1.save()
    assert p.exists()

    hits_before = bm1.query("Person1", k=5)

    bm2 = BM25Store(p)
    loaded = bm2.load()
    assert loaded is True
    assert bm2.size == 5
    assert set(bm2._ids) == set(bm1._ids)

    hits_after = bm2.query("Person1", k=5)
    # Same scores after reload
    assert hits_before == hits_after


def test_bibliography_label_match():
    """The motivating case: dense embedding misses 'Last, First' style author
    labels; BM25 must find them via lexical token overlap."""
    nodes = [
        _mk_node("Stekoll, M.", summary="seaweed researcher"),
        _mk_node("Rod Fujita", summary="environmental scientist"),
        _mk_node("Fujita, R.", summary="kelp study author"),
        _mk_node("McKinley Research Group", type_="Organization",
                 summary="market analysis firm"),
    ]
    gs = _mk_storage(nodes)
    bm = BM25Store("/dev/null/bm25.pkl")
    bm.fit_from(gs)

    # Query for "M. Stekoll" should rank Stekoll, M. highly
    hits = bm.query("M. Stekoll", k=4)
    top_id = hits[0][0]
    top_node = gs.get_node(top_id)
    assert "stekoll" in top_node.label.lower()

    # Query for "R. Fujita" should rank a Fujita node first (either form OK)
    hits = bm.query("R. Fujita", k=4)
    top_id = hits[0][0]
    top_node = gs.get_node(top_id)
    assert "fujita" in top_node.label.lower()


def test_load_missing_file_returns_false():
    bm = BM25Store("/nonexistent/path/bm25.pkl")
    assert bm.load() is False
    assert bm.size == 0
