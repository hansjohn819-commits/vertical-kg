"""Unit tests for the 4e ontology_compliance sleep-pass sub-op (§16.19 c).

Covers each of the five decision-tree branches in isolation, then a
combined sweep to verify branches don't cascade.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.graph.models import DirectProv, Edge, Node
from src.graph.storage import GraphStorage
from src.modules.m4_sleep_pass.ontology_compliance import (
    enforce_ontology_compliance,
)


def _mk_storage(tmp_path: Path) -> GraphStorage:
    return GraphStorage(tmp_path / "graph.pkl")


def _add_node(gs: GraphStorage, label: str, type_: str) -> Node:
    n = Node(
        id=str(uuid4()),
        type=type_,
        label=label,
        summary=f"{label} is a {type_}",
        provenance=DirectProv(raw_doc_id="t.pdf", extraction_run_id="r0"),
    )
    gs.add_node(n)
    return n


def _add_edge(gs: GraphStorage, src: Node, tgt: Node, type_: str,
              original_type: str | None = None) -> Edge:
    e = Edge(
        id=str(uuid4()),
        source_id=src.id,
        target_id=tgt.id,
        type=type_,
        provenance=DirectProv(raw_doc_id="t.pdf", extraction_run_id="r0"),
        original_type=original_type,
    )
    gs.add_edge(e)
    return e


# Minimal ontology fixture used by every test below.
ONTOLOGY_RELS = {"AUTHORED_BY", "LOCATED_IN", "RELATED_TO"}
DOMAIN_MAP = {
    "AUTHORED_BY": ({"Publication"}, {"Person"}),
    "LOCATED_IN":  ({"Company", "Person"}, {"Location"}),
}
ALIASES = {"AUTHED_BY": "AUTHORED_BY"}


def test_branch1_alias_rewrites_type(tmp_path):
    gs = _mk_storage(tmp_path)
    src = _add_node(gs, "Paper A", "Publication")
    tgt = _add_node(gs, "Alice", "Person")
    e = _add_edge(gs, src, tgt, "AUTHED_BY")  # alias of AUTHORED_BY

    counters = enforce_ontology_compliance(gs, ONTOLOGY_RELS, DOMAIN_MAP, ALIASES)

    assert counters["alias_rewrites"] == 1
    assert gs.get_edge(e.id).type == "AUTHORED_BY"
    assert gs.get_edge(e.id).original_type is None  # alias rewrite ≠ downgrade


def test_branch2_registered_ok_is_noop(tmp_path):
    gs = _mk_storage(tmp_path)
    src = _add_node(gs, "Paper A", "Publication")
    tgt = _add_node(gs, "Alice", "Person")
    e = _add_edge(gs, src, tgt, "AUTHORED_BY")

    counters = enforce_ontology_compliance(gs, ONTOLOGY_RELS, DOMAIN_MAP, ALIASES)

    assert sum(v for k, v in counters.items() if k != "examined") == 0
    assert gs.get_edge(e.id).type == "AUTHORED_BY"


def test_branch3_domain_mismatch_downgrades_and_stashes(tmp_path):
    gs = _mk_storage(tmp_path)
    # AUTHORED_BY's declared domain is (Publication, Person); using it on
    # (Person, Person) violates the source-type constraint.
    src = _add_node(gs, "Bob", "Person")
    tgt = _add_node(gs, "Alice", "Person")
    e = _add_edge(gs, src, tgt, "AUTHORED_BY")

    counters = enforce_ontology_compliance(gs, ONTOLOGY_RELS, DOMAIN_MAP, ALIASES)

    assert counters["domain_downgrades"] == 1
    assert gs.get_edge(e.id).type == "RELATED_TO"
    assert gs.get_edge(e.id).original_type == "AUTHORED_BY"


def test_branch4_unregistered_type_downgrades_and_stashes(tmp_path):
    gs = _mk_storage(tmp_path)
    src = _add_node(gs, "Bob", "Person")
    tgt = _add_node(gs, "Maine", "Location")
    e = _add_edge(gs, src, tgt, "COMPARED_TO")  # not in ontology

    counters = enforce_ontology_compliance(gs, ONTOLOGY_RELS, DOMAIN_MAP, ALIASES)

    assert counters["noise_downgrades"] == 1
    assert gs.get_edge(e.id).type == "RELATED_TO"
    assert gs.get_edge(e.id).original_type == "COMPARED_TO"


def test_branch5_rescue_restores_when_ontology_catches_up(tmp_path):
    gs = _mk_storage(tmp_path)
    # An edge previously downgraded — type=RELATED_TO, original_type
    # records the proposal. Ontology now includes that type and the
    # domain fits → 4e should restore it.
    src = _add_node(gs, "Bob", "Person")
    tgt = _add_node(gs, "Maine", "Location")
    e = _add_edge(gs, src, tgt, "RELATED_TO", original_type="LOCATED_IN")

    counters = enforce_ontology_compliance(gs, ONTOLOGY_RELS, DOMAIN_MAP, ALIASES)

    assert counters["rescues"] == 1
    edge = gs.get_edge(e.id)
    assert edge.type == "LOCATED_IN"
    # Branch 5 explicitly preserves original_type as audit history.
    assert edge.original_type == "LOCATED_IN"


def test_branch5_no_rescue_when_original_type_still_violates(tmp_path):
    gs = _mk_storage(tmp_path)
    # original_type=AUTHORED_BY but src is Person (not Publication) →
    # domain still fails → leave as RELATED_TO.
    src = _add_node(gs, "Bob", "Person")
    tgt = _add_node(gs, "Alice", "Person")
    e = _add_edge(gs, src, tgt, "RELATED_TO", original_type="AUTHORED_BY")

    counters = enforce_ontology_compliance(gs, ONTOLOGY_RELS, DOMAIN_MAP, ALIASES)

    assert counters["rescues"] == 0
    edge = gs.get_edge(e.id)
    assert edge.type == "RELATED_TO"
    assert edge.original_type == "AUTHORED_BY"  # unchanged


def test_branch5_no_rescue_when_original_type_left_ontology(tmp_path):
    gs = _mk_storage(tmp_path)
    # original_type refers to a type that's no longer in ontology
    # (e.g., the user removed it during a revision).
    src = _add_node(gs, "Bob", "Person")
    tgt = _add_node(gs, "Maine", "Location")
    e = _add_edge(gs, src, tgt, "RELATED_TO", original_type="LIVES_NEAR")

    counters = enforce_ontology_compliance(gs, ONTOLOGY_RELS, DOMAIN_MAP, ALIASES)

    assert counters["rescues"] == 0
    assert gs.get_edge(e.id).type == "RELATED_TO"


def test_legitimate_related_to_left_alone(tmp_path):
    gs = _mk_storage(tmp_path)
    # RELATED_TO with no stashed original_type — model genuinely emitted
    # RELATED_TO (or M1 wrote it directly). Nothing to do.
    src = _add_node(gs, "Bob", "Person")
    tgt = _add_node(gs, "Alice", "Person")
    e = _add_edge(gs, src, tgt, "RELATED_TO")

    counters = enforce_ontology_compliance(gs, ONTOLOGY_RELS, DOMAIN_MAP, ALIASES)

    assert sum(v for k, v in counters.items() if k != "examined") == 0
    assert gs.get_edge(e.id).type == "RELATED_TO"
    assert gs.get_edge(e.id).original_type is None


def test_mixed_sweep_branches_dont_cascade(tmp_path):
    """One pass per edge — branch 1's alias rewrite does NOT then fall
    through into branch 3 even if the resolved type happens to violate
    domain. The next pass would catch it."""
    gs = _mk_storage(tmp_path)
    src = _add_node(gs, "Bob", "Person")
    tgt = _add_node(gs, "Alice", "Person")
    # Alias resolves AUTHED_BY → AUTHORED_BY, which violates domain for
    # (Person, Person). Branch 1 fires; branch 3 deliberately doesn't.
    e = _add_edge(gs, src, tgt, "AUTHED_BY")

    counters = enforce_ontology_compliance(gs, ONTOLOGY_RELS, DOMAIN_MAP, ALIASES)

    assert counters["alias_rewrites"] == 1
    assert counters["domain_downgrades"] == 0
    edge = gs.get_edge(e.id)
    assert edge.type == "AUTHORED_BY"  # alias-resolved, sits in registered+bad-domain state
    # Second pass picks it up via branch 3.
    counters2 = enforce_ontology_compliance(gs, ONTOLOGY_RELS, DOMAIN_MAP, ALIASES)
    assert counters2["domain_downgrades"] == 1
    edge2 = gs.get_edge(e.id)
    assert edge2.type == "RELATED_TO"
    assert edge2.original_type == "AUTHORED_BY"
