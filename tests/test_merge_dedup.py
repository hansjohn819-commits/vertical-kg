"""Unit tests for the 4b merge-step speedups (commit topic: 4b 0-merge
short-circuit + per-pass rejected-pair memory).

These exercise the state plumbing only — `_judge_pair` is mocked so no
LLM is needed. The point is to prove:

  * A round with 0 merges short-circuits `_vote_done` (no extra LLM
    call) and sets `should_stop = True`.
  * Pairs that landed in `merge_rejected_pairs` on a prior round get
    skipped on the next round instead of re-asking the judge.
  * `new_rejections` accumulates only "different" verdicts, mirroring
    the merge_reject log emission.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

from src.graph.models import DirectProv, Node
from src.graph.storage import GraphStorage
from src.modules.m4_sleep_pass import merge as merge_mod


# --- Fixtures ----------------------------------------------------------

def _node(label: str, type_: str = "Concept") -> Node:
    return Node(
        id=str(uuid4()),
        type=type_,
        label=label,
        summary=f"{label} description",
        provenance=DirectProv(raw_doc_id="t.pdf", extraction_run_id="r0"),
    )


class _StubVectorStore:
    """Minimal stand-in. `_candidate_pairs` is patched out, so the real
    vector store is never touched."""

    def remove(self, _id): pass


class _StubInstance:
    def __init__(self, storage):
        self.storage = storage
        self.vector_store = _StubVectorStore()


def _mk_storage(tmp_path) -> GraphStorage:
    return GraphStorage(tmp_path / "graph.pkl")


# --- Tests -------------------------------------------------------------

def test_zero_merge_short_circuits_vote_done(tmp_path):
    """When the judge says 'different' on everything, the round should
    stop without consulting _vote_done — that's where the 18 min
    duplicate-round waste came from (2026-05-17 pass)."""
    gs = _mk_storage(tmp_path)
    a, b = _node("A"), _node("B")
    gs.add_node(a)
    gs.add_node(b)
    state = {
        "pass_id": "test", "merge_iter": 0,
        "merge_rejected_pairs": [], "stats": {},
    }

    judge_calls = []
    def fake_judge(_client, _storage, na, nb):
        judge_calls.append((na.id, nb.id))
        return {"verdict": "different", "why": "they aren't"}

    vote_calls = []
    def fake_vote(_client, _summary):
        vote_calls.append(_summary)
        return False  # would normally force another round

    with patch.object(merge_mod, "_candidate_pairs",
                      return_value=[(a, b, 0.9)]), \
         patch.object(merge_mod, "_judge_pair", side_effect=fake_judge), \
         patch.object(merge_mod, "_vote_done", side_effect=fake_vote), \
         patch.object(merge_mod, "get_client", return_value=None):
        result = merge_mod.merge_step(state, instance=_StubInstance(gs))

    assert result["merge_done_vote"] is True
    assert len(judge_calls) == 1
    assert vote_calls == []  # short-circuited
    assert "|".join(sorted([a.id, b.id])) in result["merge_rejected_pairs"]


def test_rejected_pair_skipped_on_next_round(tmp_path):
    """A pair the judge rejected in round 1 must not get re-asked in
    round 2 — that's how a single judgement set of 200 pairs used to
    burn 2 rounds × 18 min."""
    gs = _mk_storage(tmp_path)
    a, b = _node("A"), _node("B")
    c, d = _node("C"), _node("D")  # second pair, fresh
    for n in (a, b, c, d):
        gs.add_node(n)

    rejected_key = "|".join(sorted([a.id, b.id]))
    state = {
        "pass_id": "test", "merge_iter": 1,
        "merge_rejected_pairs": [rejected_key],
        "stats": {},
    }

    judge_calls = []
    def fake_judge(_client, _storage, na, nb):
        judge_calls.append((na.id, nb.id))
        return {"verdict": "different", "why": "nope"}

    with patch.object(merge_mod, "_candidate_pairs",
                      return_value=[(a, b, 0.9), (c, d, 0.88)]), \
         patch.object(merge_mod, "_judge_pair", side_effect=fake_judge), \
         patch.object(merge_mod, "_vote_done", return_value=True), \
         patch.object(merge_mod, "get_client", return_value=None):
        result = merge_mod.merge_step(state, instance=_StubInstance(gs))

    # Only the fresh pair (c, d) should have been judged. The old (a, b)
    # is in rejected memory.
    assert len(judge_calls) == 1
    assert judge_calls[0] == (c.id, d.id)
    # Round stats records the skip count.
    round_stats = result["stats"]["merge_round_1"]
    assert round_stats["skipped_already_rejected"] == 1


def test_caveat_does_not_pollute_rejection_memory(tmp_path):
    """`same_with_caveats` is not a 'different' verdict — it shouldn't
    end up in rejected_pairs, so the same pair can be revisited next
    pass after caveat-driven weight decay shifts the picture."""
    gs = _mk_storage(tmp_path)
    a, b = _node("A"), _node("B")
    gs.add_node(a)
    gs.add_node(b)
    state = {
        "pass_id": "test", "merge_iter": 0,
        "merge_rejected_pairs": [], "stats": {},
    }

    def fake_judge(_c, _s, _na, _nb):
        return {"verdict": "same_with_caveats", "why": "maybe"}

    with patch.object(merge_mod, "_candidate_pairs",
                      return_value=[(a, b, 0.9)]), \
         patch.object(merge_mod, "_judge_pair", side_effect=fake_judge), \
         patch.object(merge_mod, "_vote_done", return_value=True), \
         patch.object(merge_mod, "get_client", return_value=None):
        result = merge_mod.merge_step(state, instance=_StubInstance(gs))

    assert result["merge_rejected_pairs"] == []
