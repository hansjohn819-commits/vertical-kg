"""Tests for the §16.17 source-text linkage layer.

Covers the pure-Python pieces that don't need a live LLM backend:
TextUnitStore CRUD + persistence, the storage backfill that adds new
schema fields to old pickled graphs, the boot-time consistency check
in GraphInstance, the M1 quote-matching helpers, and the
chunk-aware evidence builder used by graph_query / fast_query.

Run: ``python -m pytest tests/test_text_units.py -v``
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.graph.instance import GraphInstance
from src.graph.models import DirectProv, Edge, Node
from src.graph.storage import GraphStorage
from src.graph.text_units import TextUnitStore


# ---------------------------------------------------------------------------
# TextUnitStore
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store(tmp_path: Path) -> TextUnitStore:
    return TextUnitStore(tmp_path / "text_units.json")


def _sample_unit(**overrides) -> dict:
    base = {
        "id": uuid4().hex,
        "text": "Sample chunk text.",
        "raw_doc_id": "doc.pdf",
        "page_num": 1,
        "n_tokens": 4,
        "created_at": "2026-05-08T20:00:00Z",
    }
    base.update(overrides)
    return base


def test_add_and_get_roundtrip(tmp_store: TextUnitStore) -> None:
    u = _sample_unit()
    tmp_store.add(u)
    got = tmp_store.get(u["id"])
    assert got == u
    assert u["id"] in tmp_store
    assert len(tmp_store) == 1


def test_get_missing_returns_none(tmp_store: TextUnitStore) -> None:
    assert tmp_store.get("nonexistent") is None


def test_get_many_skips_missing_silently(tmp_store: TextUnitStore) -> None:
    a = _sample_unit(text="A")
    b = _sample_unit(text="B")
    tmp_store.add(a)
    tmp_store.add(b)
    out = tmp_store.get_many([a["id"], "missing", b["id"]])
    # Order preserved, missing dropped silently.
    assert [u["text"] for u in out] == ["A", "B"]


def test_remove_idempotent(tmp_store: TextUnitStore) -> None:
    u = _sample_unit()
    tmp_store.add(u)
    assert tmp_store.remove(u["id"]) is True
    assert tmp_store.remove(u["id"]) is False
    assert u["id"] not in tmp_store


def test_save_and_load_atomic(tmp_path: Path) -> None:
    store_a = TextUnitStore(tmp_path / "tu.json")
    u = _sample_unit(text="Persistence test.")
    store_a.add(u)
    store_a.save()
    # Verify on-disk shape is valid JSON keyed by chunk_id.
    raw = json.loads((tmp_path / "tu.json").read_text(encoding="utf-8"))
    assert u["id"] in raw
    # Fresh store sees what the previous one wrote.
    store_b = TextUnitStore(tmp_path / "tu.json")
    assert store_b.load() is True
    assert store_b.get(u["id"]) == u


def test_load_missing_file_returns_false(tmp_path: Path) -> None:
    store = TextUnitStore(tmp_path / "does_not_exist.json")
    assert store.load() is False
    assert len(store) == 0


def test_by_raw_doc_sorts_by_page(tmp_store: TextUnitStore) -> None:
    tmp_store.add(_sample_unit(raw_doc_id="a.pdf", page_num=3))
    tmp_store.add(_sample_unit(raw_doc_id="a.pdf", page_num=1))
    tmp_store.add(_sample_unit(raw_doc_id="a.pdf", page_num=2))
    tmp_store.add(_sample_unit(raw_doc_id="other.pdf", page_num=1))
    pages = [u["page_num"] for u in tmp_store.by_raw_doc("a.pdf")]
    assert pages == [1, 2, 3]
    assert len(tmp_store.by_raw_doc("missing.pdf")) == 0


# ---------------------------------------------------------------------------
# Schema backfill (storage.load → _backfill_new_fields)
# ---------------------------------------------------------------------------


def _make_node(label: str = "X", **overrides) -> Node:
    return Node(
        type="Entity",
        label=label,
        provenance=DirectProv(raw_doc_id="d.pdf", extraction_run_id="r1"),
        **overrides,
    )


def _make_edge(src: str, tgt: str, **overrides) -> Edge:
    return Edge(
        source_id=src, target_id=tgt, type="RELATED_TO",
        provenance=DirectProv(raw_doc_id="d.pdf", extraction_run_id="r1"),
        **overrides,
    )


def test_backfill_adds_text_unit_ids_to_legacy_node(tmp_path: Path) -> None:
    # Simulate a node pickled before §16.17 by stripping the new field
    # from the model instance's __dict__.
    n = _make_node("Legacy")
    object.__delattr__(n, "text_unit_ids") if hasattr(n, "text_unit_ids") else None
    if "text_unit_ids" in n.__dict__:
        del n.__dict__["text_unit_ids"]
    storage = GraphStorage(tmp_path / "graph.pkl")
    storage._g.add_node(n.id, data=n)
    storage.save()
    # Reload — the backfill in storage.load() should re-instantiate via
    # pydantic and the new field becomes accessible with empty-list default.
    fresh = GraphStorage(tmp_path / "graph.pkl")
    fresh.load()
    reloaded = fresh.get_node(n.id)
    assert reloaded is not None
    assert reloaded.text_unit_ids == []


def test_backfill_adds_evidence_quote_to_legacy_edge(tmp_path: Path) -> None:
    a = _make_node("A")
    b = _make_node("B")
    e = _make_edge(a.id, b.id)
    if "text_unit_ids" in e.__dict__:
        del e.__dict__["text_unit_ids"]
    if "evidence_quote" in e.__dict__:
        del e.__dict__["evidence_quote"]
    storage = GraphStorage(tmp_path / "graph.pkl")
    storage.add_node(a)
    storage.add_node(b)
    storage._g.add_edge(a.id, b.id, data=e)
    storage.save()
    fresh = GraphStorage(tmp_path / "graph.pkl")
    fresh.load()
    edges = list(fresh.edges())
    assert len(edges) == 1
    assert edges[0].text_unit_ids == []
    assert edges[0].evidence_quote == ""


# ---------------------------------------------------------------------------
# GraphInstance boot consistency check
# ---------------------------------------------------------------------------


def _new_instance(tmp_path: Path) -> GraphInstance:
    """Build a fresh GraphInstance backed by tmp_path. Also creates a stub
    ontology.md so the instance loads cleanly."""
    ont = tmp_path / "ontology.md"
    ont.write_text("# Entity Types\n\n# Relation Types\n", encoding="utf-8")
    return GraphInstance(
        name="test", storage_path=tmp_path, ontology_path=ont,
    )


def test_consistency_keeps_unreferenced_chunk(tmp_path: Path) -> None:
    """Boot consistency must NOT delete unreferenced chunks (revised
    behavior after the 2026-05-08 incident where a mid-PASS-1 ingest
    crash left chunks for not-yet-processed pages with no node refs;
    the old logic deleted them, destroying the only on-disk copy of
    page text and blocking any subsequent retry from continuing where
    the failed run left off).
    """
    gi = _new_instance(tmp_path)
    cid = uuid4().hex
    gi.text_units.add(_sample_unit(id=cid))
    gi.save()

    gi2 = _new_instance(tmp_path)
    assert cid in gi2.text_units, "unreferenced chunk must survive boot"


def test_consistency_scrubs_dangling_chunk_id_from_node(tmp_path: Path) -> None:
    gi = _new_instance(tmp_path)
    # Node references a chunk_id that doesn't exist in text_units.
    n = _make_node("Dangling")
    n.text_unit_ids = ["nonexistent-chunk"]
    gi.storage.add_node(n)
    gi.save()

    gi2 = _new_instance(tmp_path)
    reloaded = gi2.storage.get_node(n.id)
    assert reloaded is not None
    assert reloaded.text_unit_ids == []


def test_consistency_keeps_valid_pairings(tmp_path: Path) -> None:
    gi = _new_instance(tmp_path)
    cid = uuid4().hex
    gi.text_units.add(_sample_unit(id=cid))
    n = _make_node("Anchored")
    n.text_unit_ids = [cid]
    gi.storage.add_node(n)
    gi.save()

    gi2 = _new_instance(tmp_path)
    assert cid in gi2.text_units
    assert gi2.storage.get_node(n.id).text_unit_ids == [cid]


# ---------------------------------------------------------------------------
# M1 pure-Python helpers
# ---------------------------------------------------------------------------


def test_quote_matches_basic() -> None:
    from src.modules.m1_ingest import _quote_matches

    page = "Atlantic Sea Farms processed 1.2 million pounds of kelp in 2024."
    assert _quote_matches("Atlantic Sea Farms processed 1.2 million pounds", page)
    # Case-insensitive
    assert _quote_matches("ATLANTIC sea FARMS PROCESSED", page)
    # Punctuation differences tolerated
    assert _quote_matches("Atlantic Sea Farms, processed 1.2 million", page)
    # Wholly absent
    assert not _quote_matches("Maine has tax credits", page)
    # Empty / too short
    assert not _quote_matches("", page)
    assert not _quote_matches("ASF", page)  # < 8 normalized chars


def test_format_prior_entities_empty_and_populated() -> None:
    from src.modules.m1_ingest import _format_prior_entities

    out = _format_prior_entities([])
    assert "(none" in out

    n = _make_node("ASF")
    n.summary = "A kelp cooperative founded in 2010"
    out = _format_prior_entities([n])
    assert "ASF" in out
    assert "Entity" in out  # type rendered
    assert "kelp cooperative" in out


def test_split_ontology_yields_relation_block() -> None:
    from src.modules.m1_ingest import _split_ontology

    ont = (
        "# Entity Types\n## Company\n  required: [label]\n\n"
        "# Relation Types\n## CEO_OF\n  domain: Person × Company\n\n"
        "# Global Conventions\n- summary: 100 tokens\n"
    )
    full, rel = _split_ontology(ont)
    assert "Entity Types" in full
    assert "CEO_OF" in rel
    assert "Global Conventions" not in rel  # trimmed at boundary
    assert "Entity Types" not in rel  # only relations after the split


def test_split_ontology_handles_missing_section() -> None:
    from src.modules.m1_ingest import _split_ontology

    full, rel = _split_ontology("just text, no sections")
    assert "no relation types section" in rel.lower()


# ---------------------------------------------------------------------------
# Chunk-aware evidence builder
# ---------------------------------------------------------------------------


def test_evidence_builder_renders_chunks_when_present(tmp_path: Path) -> None:
    from src.modules.m2_qa_agent import _build_chunk_evidence

    gi = _new_instance(tmp_path)
    cid = uuid4().hex
    gi.text_units.add(_sample_unit(
        id=cid,
        text="ASF processed 1.2 million pounds of kelp in 2024.",
        raw_doc_id="ASF Impact Report 2024.pdf",
        page_num=5,
    ))
    n = _make_node("ASF")
    n.type = "Organization"
    n.summary = "A kelp cooperative."
    n.text_unit_ids = [cid]
    gi.storage.add_node(n)

    ctx, ids, chunk_ids = _build_chunk_evidence(
        gi, [n],
        question="How much kelp did ASF process in 2024?",
        include_neighbors=False, use_display_title=True,
    )
    assert "Entity: ASF (Organization)" in ctx
    assert "Summary: A kelp cooperative." in ctx
    assert "Sources:" in ctx
    # display_title strips the .pdf suffix
    assert "ASF Impact Report 2024" in ctx
    assert ".pdf" not in ctx
    assert "page 5" in ctx
    assert "ASF processed 1.2 million pounds" in ctx
    assert n.id in ids
    assert cid in chunk_ids


def test_evidence_builder_legacy_node_no_chunks(tmp_path: Path) -> None:
    """Old graphs without text_unit_ids still produce a clean block (just
    Entity + Summary, no Sources section)."""
    from src.modules.m2_qa_agent import _build_chunk_evidence

    gi = _new_instance(tmp_path)
    n = _make_node("LegacyEntity")
    n.summary = "summary only"
    gi.storage.add_node(n)
    ctx, ids, chunk_ids = _build_chunk_evidence(
        gi, [n],
        question="anything",
        include_neighbors=False, use_display_title=True,
    )
    assert "LegacyEntity" in ctx
    assert "summary only" in ctx
    assert "Sources:" not in ctx
    assert n.id in ids
    assert chunk_ids == []


def test_evidence_builder_neighbors_mode(tmp_path: Path) -> None:
    from src.modules.m2_qa_agent import _build_chunk_evidence

    gi = _new_instance(tmp_path)
    a = _make_node("A")
    b = _make_node("B")
    a.summary = "seed"
    b.summary = "neighbor"
    gi.storage.add_node(a)
    gi.storage.add_node(b)
    gi.storage.add_edge(_make_edge(a.id, b.id))

    ctx, ids, _chunk_ids = _build_chunk_evidence(
        gi, [a],
        question="A and B",
        include_neighbors=True, use_display_title=False,
    )
    assert "Entity: A" in ctx
    assert "NEIGHBORS" in ctx
    assert "neighbor" in ctx  # B's summary
    assert {a.id, b.id}.issubset(ids)


def test_evidence_builder_external_omits_neighbors(tmp_path: Path) -> None:
    from src.modules.m2_qa_agent import _build_chunk_evidence

    gi = _new_instance(tmp_path)
    a = _make_node("A")
    b = _make_node("B")
    gi.storage.add_node(a)
    gi.storage.add_node(b)
    gi.storage.add_edge(_make_edge(a.id, b.id))

    ctx, _, _ = _build_chunk_evidence(
        gi, [a],
        question="anything",
        include_neighbors=False, use_display_title=True,
    )
    assert "NEIGHBORS" not in ctx


# ---------------------------------------------------------------------------
# show_provenance edge support
# ---------------------------------------------------------------------------


def test_show_provenance_edge(tmp_path: Path) -> None:
    """The §16.8.4 / §16.17 extension: show_provenance resolves edges too,
    returning source/target labels, evidence_quote, and any text_unit_ids
    resolved back to their chunks.

    Post-2026-05-15 router refactor: show_provenance is now a Python API
    on GraphInstance (no longer LLM-callable from chat). Tests against
    instance.show_provenance() directly.
    """
    gi = _new_instance(tmp_path)
    cid = uuid4().hex
    gi.text_units.add(_sample_unit(
        id=cid,
        text="A is the parent of B.",
        raw_doc_id="rel.pdf",
        page_num=2,
    ))
    a = _make_node("A")
    b = _make_node("B")
    gi.storage.add_node(a)
    gi.storage.add_node(b)
    e = _make_edge(a.id, b.id)
    e.text_unit_ids = [cid]
    e.evidence_quote = "A is the parent of B."
    gi.storage._g.add_edge(a.id, b.id, data=e)

    res = gi.show_provenance(e.id)
    assert res["kind"] == "edge"
    assert res["source"]["label"] == "A"
    assert res["target"]["label"] == "B"
    assert res["evidence_quote"] == "A is the parent of B."
    assert len(res["sources"]) == 1
    assert res["sources"][0]["page_num"] == 2
    assert "A is the parent of B." in res["sources"][0]["text"]


def test_ingest_resilient_to_per_page_llm_exception(
    tmp_path: Path, monkeypatch
) -> None:
    """Page-level LLM failures must not abort the whole document. PASS 1
    page 2 raises; pages 1 and 3 succeed → document completes with
    nodes from pages 1 and 3, page 2 logged as a warning, all chunks
    persisted on disk (atomic save called per page)."""
    from src.modules import m1_ingest as m1
    import json

    # Track which pages have been called; raise once on page 2.
    call_count = {"n": 0}

    def fake_llm_with_retry(client, **kwargs):
        call_count["n"] += 1
        page_num = kwargs.get("page_num")
        if page_num == 2:
            raise TimeoutError("simulated request timeout")
        # Pretend pass1 returns one entity per page; PASS 2 returns [].
        if kwargs.get("expect") == "object":
            label = f"Entity_p{page_num}"
            return (
                {
                    "entities": [{
                        "label": label, "type": "Entity",
                        "summary": f"summary {page_num}",
                        "source_quote": "",
                    }],
                    "relationships": [],
                },
                None, "stop",
            )
        else:  # PASS 2
            return [], None, "stop"

    monkeypatch.setattr(m1, "_llm_with_retry", fake_llm_with_retry)

    # Stub out get_client so we don't actually hit the LLM backend.
    class _FakeClient: pass
    monkeypatch.setattr(m1, "get_client", lambda role="backend": _FakeClient())

    gi = _new_instance(tmp_path)
    result = m1.ingest_document(
        storage=gi.storage,
        vector_store=None,  # skip vector encoding so the test stays pure-Python
        text_units=gi.text_units,
        ontology_path=None,
        raw_doc_id="doc.pdf",
        pages=["page one text", "page two text", "page three text"],
    )

    # Pages 1 and 3 produced one entity each; page 2 raised.
    assert result.nodes_added == 2
    assert any(
        w["page_num"] == 2 and w["pass"] == "pass1"
        and w["warning"] == "llm_call_exception"
        for w in result.page_warnings
    )

    # All 3 chunks persisted on disk (text_units saved up front).
    saved = json.loads((tmp_path / "text_units.json").read_text(encoding="utf-8"))
    assert len(saved) == 3, "all chunks must persist even when a page fails"


def test_show_provenance_not_found(tmp_path: Path) -> None:
    gi = _new_instance(tmp_path)
    assert gi.show_provenance("nope") == {"kind": "not_found", "id": "nope"}


# ---------------------------------------------------------------------------
# Ontology parser + domain validator (§16.17 problem 4)
# ---------------------------------------------------------------------------


def test_parse_ontology_real_file() -> None:
    """The shipped ontology.md must round-trip cleanly. Test pins the
    six seed entity types + the currently-registered relation types.
    If either set drifts further from these, this fixture catches it."""
    from src.modules.m1_ingest import parse_ontology

    ont = (Path(__file__).resolve().parent.parent / "ontology.md").read_text(encoding="utf-8")
    ents, rels, dom = parse_ontology(ont)
    assert ents == {"Company", "Person", "Product", "Location", "Organization", "Industry"}
    expected_rels = {
        # Seed types (2026-04-21)
        "CEO_OF", "FOUNDED", "CO_FOUNDED", "PRODUCES", "HEADQUARTERED_IN",
        "IN_INDUSTRY", "REGULATES", "OWNS", "RELATED_TO",
        # Round-1 additions (2026-05-09 morning)
        "AUTHORED_WITH", "AFFILIATED_WITH", "PUBLISHED_BY",
        "STUDIED_LOCATION", "LOCATED_IN", "OPERATES_IN",
        # Round-2 additions (2026-05-09 evening, ASF-driven)
        "AUDITED_BY", "AUDITS",
        # Round-3 additions (2026-05-11, FAO SOFIA + corpus-wide signals)
        "PRODUCED_BY", "PRODUCED_IN", "AUTHORED_BY", "PUBLISHED",
        "EXPORTS", "INCLUDES", "PART_OF", "FUNDED_BY",
        # Round-4 additions (2026-05-12, post round-3 corpus signals)
        "FUNDED", "IMPORTED_FROM", "PARTNERED_WITH", "CONTAINS",
        "PRODUCES_LOCATION",
        # Round-5 additions (2026-05-17, post 3 new docs)
        "HOSTS_OPERATIONS_OF",
    }
    assert rels == expected_rels
    # Spot-check domain extensions across rounds.
    assert dom["CEO_OF"] == ({"Person"}, {"Company", "Organization"})
    assert dom["AFFILIATED_WITH"] == (
        {"Person", "Organization", "Company"}, {"Organization", "Company"},
    )
    # Round-5 extended STUDIED_LOCATION src to include Industry, tgt to include Industry.
    assert dom["STUDIED_LOCATION"] == (
        {"Person", "Organization", "Product", "Industry"},
        {"Location", "Industry"},
    )
    assert dom["AUDITED_BY"] == ({"Organization", "Company"}, {"Organization", "Company"})
    assert dom["FOUNDED"] == ({"Person"}, {"Company", "Organization"})  # union
    assert dom["PRODUCES"] == ({"Company", "Organization"}, {"Product"})
    # Round-5 extended AUTHORED_WITH to (Person|Organization) × (Person|Organization).
    assert dom["AUTHORED_WITH"] == (
        {"Person", "Organization"}, {"Person", "Organization"},
    )
    assert dom["RELATED_TO"] == (set(), set())  # any × any
    # Round-3 spot checks.
    assert dom["IN_INDUSTRY"] == ({"Company", "Product", "Location"}, {"Industry"})
    assert dom["PRODUCED_BY"] == ({"Product"}, {"Company", "Organization"})
    assert dom["PRODUCED_IN"] == ({"Product"}, {"Location"})
    assert dom["PART_OF"] == (set(), set())  # any × any
    # Round-5 spot checks: AUTHORED_BY src adds Company; LOCATED_IN src adds Person + Industry.
    assert dom["AUTHORED_BY"] == (
        {"Person", "Organization", "Company"}, {"Product"},
    )
    assert dom["LOCATED_IN"] == (
        {"Person", "Organization", "Company", "Location", "Industry"},
        {"Location"},
    )
    assert dom["HOSTS_OPERATIONS_OF"] == (
        {"Location"}, {"Company", "Organization"},
    )
    assert dom["EXPORTS"] == (
        {"Company", "Organization", "Location"}, {"Product", "Location"},
    )
    assert dom["IMPORTED_FROM"] == (
        {"Company", "Organization", "Location"},
        {"Company", "Organization", "Location", "Product"},
    )
    assert dom["PARTNERED_WITH"] == (
        {"Person", "Organization", "Company", "Product"},
        {"Organization", "Company", "Product"},
    )


def test_parse_aliases_real_file() -> None:
    """The shipped ontology.md ships with two known model-typo aliases.
    Parser must surface both so the M1 ingest pipeline rewrites them
    before validation. Both target AFFILIATED_WITH (the canonical
    spelling)."""
    from src.modules.m1_ingest import parse_aliases

    ont = (Path(__file__).resolve().parent.parent / "ontology.md").read_text(encoding="utf-8")
    aliases = parse_aliases(ont)
    assert aliases == {
        "AFFULIATED_WITH": "AFFILIATED_WITH",
        "AFFILIATIED_WITH": "AFFILIATED_WITH",
        # Round-3 addition (2026-05-11): bare AUTHORED collapses into the
        # newly-introduced AUTHORED_BY (person/org → publication).
        "AUTHORED": "AUTHORED_BY",
        # Round-4 additions (2026-05-12): surface variants of two newly-
        # promoted relations.
        "IMPORT_FROM": "IMPORTED_FROM",
        "PARTNERS_WITH": "PARTNERED_WITH",
        # Round-5 addition (2026-05-17): typo from the OHS document.
        "OPERATESS_IN": "OPERATES_IN",
    }


def test_parse_aliases_handles_missing_section() -> None:
    from src.modules.m1_ingest import parse_aliases

    assert parse_aliases("") == {}
    assert parse_aliases("# Entity Types\n\n# Relation Types\n") == {}


def test_parse_aliases_supports_ascii_arrow() -> None:
    from src.modules.m1_ingest import parse_aliases

    ont = (
        "# Aliases\n"
        "- FOO -> BAR\n"
        "- BAZ → QUUX\n"
        "# Global Conventions\n"
    )
    assert parse_aliases(ont) == {"FOO": "BAR", "BAZ": "QUUX"}


def test_plan_superchunks_basic_sizes() -> None:
    """The super-chunk splitter (2026-05-10) wraps M1 ingest for long
    PDFs. Pin the exact page-range layout so the tokenizer math we
    used to size the super-chunk window stays valid."""
    from src.modules.m2_qa_agent import GraphAgent

    # Single chunk when at-or-below threshold.
    assert GraphAgent._plan_superchunks(80, 80, 1) == [(0, 80)]
    assert GraphAgent._plan_superchunks(50, 80, 1) == [(0, 50)]
    # One page over threshold → split with 1-page overlap on the boundary.
    assert GraphAgent._plan_superchunks(81, 80, 1) == [(0, 80), (79, 81)]
    # 200-page document → three super-chunks, each ≤ 80 pages, 1-page overlaps.
    assert GraphAgent._plan_superchunks(200, 80, 1) == [
        (0, 80), (79, 159), (158, 200),
    ]
    # Edge cases.
    assert GraphAgent._plan_superchunks(0, 80, 1) == []
    assert GraphAgent._plan_superchunks(1, 80, 1) == [(0, 1)]


def test_plan_superchunks_no_overlap() -> None:
    """Overlap=0 still works — produces exact non-overlapping slices."""
    from src.modules.m2_qa_agent import GraphAgent

    assert GraphAgent._plan_superchunks(160, 80, 0) == [(0, 80), (80, 160)]
    assert GraphAgent._plan_superchunks(81, 80, 0) == [(0, 80), (80, 81)]


def test_plan_superchunks_covers_every_page() -> None:
    """Every page index in [0, n) must appear in at least one super-chunk —
    silent skips would lose source material."""
    from src.modules.m2_qa_agent import GraphAgent

    for n in (80, 81, 100, 159, 160, 200, 264, 500):
        ranges = GraphAgent._plan_superchunks(n, 80, 1)
        seen = set()
        for s, e in ranges:
            seen.update(range(s, e))
        assert seen == set(range(n)), f"n={n}: missing pages {set(range(n)) - seen}"


def test_m4b_merge_twin_absorbs_text_unit_ids(tmp_path: Path) -> None:
    """When M4b merges two nodes A and B that both have a same-type edge
    to a shared third node X, the fused node ends up with one twin edge
    (weight summed). Pre-2026-05-10 the absorbed edge's text_unit_ids
    were dropped — losing the source-text linkage from one input.
    Regression check: the surviving twin must hold the union of both.

    We exercise the merge primitives directly rather than running a full
    sleep pass — the goal is to pin the twin-merge path, not to test
    the M4b judge / vector store / save flow. We also bypass the
    vector store entirely by passing a stub that records add/remove
    calls so the test stays pure-Python.
    """
    from src.graph.models import DerivedProv, NodeRef
    from src.modules.m4_sleep_pass.merge import _execute_merge

    gi = _new_instance(tmp_path)

    # Two source nodes A and B that will be merged, plus a shared
    # neighbor X. A→X and B→X both exist with the same type, each
    # with its own text_unit_ids.
    a = _make_node("A")
    b = _make_node("B")
    x = _make_node("X")
    gi.storage.add_node(a)
    gi.storage.add_node(b)
    gi.storage.add_node(x)

    e_a_x = _make_edge(a.id, x.id)
    e_a_x.type = "RELATED_TO"
    e_a_x.text_unit_ids = ["chunk-a"]
    e_a_x.weight = 1.0
    e_a_x.evidence_quote = "short"

    e_b_x = _make_edge(b.id, x.id)
    e_b_x.type = "RELATED_TO"
    e_b_x.text_unit_ids = ["chunk-b"]
    e_b_x.weight = 1.5
    e_b_x.evidence_quote = "this longer quote should win on length"

    gi.storage._g.add_edge(a.id, x.id, data=e_a_x)
    gi.storage._g.add_edge(b.id, x.id, data=e_b_x)

    # Stub vector store — _execute_merge calls add/remove on it.
    class _StubVS:
        def add(self, *a, **k): pass
        def remove(self, *a, **k): pass

    fused = _execute_merge(
        gi.storage, _StubVS(), a, b,
        fused_summary="A and B merged", fused_detail="",
        pass_id="test-pass",
    )

    # Walk the merged graph: there should be exactly one A∪B → X edge,
    # and it must carry both inputs' text_unit_ids and the longer quote.
    incident = [e for e in gi.storage.incident_edges(fused.id)
                if e.source_id == fused.id and e.target_id == x.id]
    assert len(incident) == 1, f"expected one twin survivor, got {len(incident)}"
    survivor = incident[0]
    assert sorted(survivor.text_unit_ids) == ["chunk-a", "chunk-b"], (
        "twin merge must union text_unit_ids from both absorbed edges"
    )
    assert survivor.weight == 2.5, "twin merge must sum weights"
    assert survivor.evidence_quote == "this longer quote should win on length", (
        "twin merge must keep the longer evidence_quote"
    )


def test_link_form_loads_ontology_context(tmp_path: Path) -> None:
    """The §16.17 round-2 fix routes M4d link_form's edges through the
    same ontology validator M1 ingest uses. This test pins the
    integration entry point: ``_load_ontology_context(instance)``
    must read the live ontology.md, parse relation types + domain map
    + alias map, and return them as a tuple link_step can consume.

    Catches: someone removing the helper, changing the return shape,
    or breaking the ontology.md path resolution from sleep-pass
    context.
    """
    from src.modules.m4_sleep_pass.link_form import _load_ontology_context

    gi = _new_instance(tmp_path)
    # Replace the stub ontology with one that has all three sections.
    Path(gi.ontology_path).write_text(
        "# Entity Types\n## Person\n  required: [label]\n\n"
        "# Relation Types\n## CEO_OF\n  domain: Person × Company\n\n"
        "# Aliases\n- AFFULIATED_WITH → AFFILIATED_WITH\n\n"
        "# Global Conventions\n",
        encoding="utf-8",
    )

    rels, dom, aliases = _load_ontology_context(gi)
    assert "CEO_OF" in rels
    assert dom["CEO_OF"] == ({"Person"}, {"Company"})
    assert aliases == {"AFFULIATED_WITH": "AFFILIATED_WITH"}


def test_classify_edge_type_resolves_alias() -> None:
    """A typo'd type that's in the alias map gets rewritten to its
    canonical form, validated, and emits a `kind=alias_resolved`
    audit event. The non-canonical form is NEVER logged as a
    `relation_type_proposed` (which would falsely suggest a real
    new type)."""
    from src.modules.m1_ingest import _classify_edge_type, parse_ontology

    ont = (Path(__file__).resolve().parent.parent / "ontology.md").read_text(encoding="utf-8")
    _, rels, dom = parse_ontology(ont)
    alias_map = {"AFFULIATED_WITH": "AFFILIATED_WITH"}

    events = []
    out = _classify_edge_type(
        "AFFULIATED_WITH", "Person", "Sarah", "Organization", "SFU",
        "She is affiliated with SFU", rels, dom, events.append,
        raw_doc_id="d.pdf", run_id="r1", page_num=1, pass_label="pass1",
        alias_map=alias_map,
    )
    assert out == "AFFILIATED_WITH"
    # Exactly one event: the alias rewrite. No relation_type_proposed.
    assert len(events) == 1
    assert events[0]["kind"] == "alias_resolved"
    assert events[0]["alias"] == "AFFULIATED_WITH"
    assert events[0]["canonical"] == "AFFILIATED_WITH"


def test_parse_ontology_handles_missing_sections() -> None:
    from src.modules.m1_ingest import parse_ontology

    ents, rels, dom = parse_ontology("just text, no headers")
    assert ents == set()
    assert rels == set()
    assert dom == {}


def test_domain_ok_accepts_within_domain() -> None:
    from src.modules.m1_ingest import _domain_ok, parse_ontology
    ont = (Path(__file__).resolve().parent.parent / "ontology.md").read_text(encoding="utf-8")
    _, _, dom = parse_ontology(ont)
    assert _domain_ok("CEO_OF", "Person", "Company", dom) is True
    assert _domain_ok("HEADQUARTERED_IN", "Organization", "Location", dom) is True
    assert _domain_ok("RELATED_TO", "Foo", "Bar", dom) is True  # any×any


def test_domain_ok_rejects_outside_domain() -> None:
    from src.modules.m1_ingest import _domain_ok, parse_ontology
    ont = (Path(__file__).resolve().parent.parent / "ontology.md").read_text(encoding="utf-8")
    _, _, dom = parse_ontology(ont)
    # The real bug from the 33-page run: IN_INDUSTRY used with Location target.
    assert _domain_ok("IN_INDUSTRY", "Industry", "Location", dom) is False
    assert _domain_ok("CEO_OF", "Organization", "Company", dom) is False  # wrong src
    # Unknown type passes (model-proposed types aren't blocked, just logged)
    assert _domain_ok("FUNDED_BY", "Person", "Organization", dom) is True


def test_classify_edge_type_logs_proposal_for_unknown_type() -> None:
    """A model-proposed type that isn't in the ontology gets through
    unchanged, but a `kind=ontology_proposal` event is recorded with
    subkind `relation_type_proposed`."""
    from src.modules.m1_ingest import _classify_edge_type, parse_ontology
    ont = (Path(__file__).resolve().parent.parent / "ontology.md").read_text(encoding="utf-8")
    _, rels, dom = parse_ontology(ont)
    # Pick a type that is unambiguously NOT in ontology.md (would-be
    # round-4 candidate). Pre-2026-05-11 the test used FUNDED_BY which
    # was promoted in round-3 and is now registered.
    assert "MENTORED_BY" not in rels

    events = []
    out = _classify_edge_type(
        "MENTORED_BY", "Person", "Sarah", "Person", "Alex",
        "Sarah was mentored by Alex during her early career...",
        rels, dom, events.append,
        raw_doc_id="doc.pdf", run_id="r1", page_num=5, pass_label="pass1",
    )
    assert out == "MENTORED_BY"
    assert len(events) == 1
    assert events[0]["kind"] == "ontology_proposal"
    assert events[0]["subkind"] == "relation_type_proposed"
    assert events[0]["proposed_type"] == "MENTORED_BY"


def test_classify_edge_type_downgrades_on_domain_mismatch() -> None:
    """The Mexico-as-Industry case from the 33-page failure: domain
    mismatch downgrades the type to RELATED_TO and logs a `domain_mismatch`
    event so the audit trail captures why."""
    from src.modules.m1_ingest import _classify_edge_type, parse_ontology
    ont = (Path(__file__).resolve().parent.parent / "ontology.md").read_text(encoding="utf-8")
    _, rels, dom = parse_ontology(ont)

    events = []
    out = _classify_edge_type(
        "IN_INDUSTRY", "Industry", "Phyconomy", "Location", "Mexico",
        "and 7 in Mexico", rels, dom, events.append,
        raw_doc_id="doc.pdf", run_id="r1", page_num=5, pass_label="pass1",
    )
    assert out == "RELATED_TO"
    assert events[0]["subkind"] == "domain_mismatch"
    assert events[0]["relation_type"] == "IN_INDUSTRY"


def test_classify_edge_type_passes_through_valid() -> None:
    from src.modules.m1_ingest import _classify_edge_type, parse_ontology
    ont = (Path(__file__).resolve().parent.parent / "ontology.md").read_text(encoding="utf-8")
    _, rels, dom = parse_ontology(ont)

    events = []
    out = _classify_edge_type(
        "CEO_OF", "Person", "Elon", "Company", "Tesla",
        "Elon Musk, CEO of Tesla...", rels, dom, events.append,
        raw_doc_id="doc.pdf", run_id="r1", page_num=1, pass_label="pass1",
    )
    assert out == "CEO_OF"
    assert events == []  # valid types don't log proposals


def test_dedupe_edges_basic(tmp_path: Path) -> None:
    """Three edges with the same (src, tgt, type) collapse to one. The
    survivor accumulates the union of text_unit_ids, sums the weights,
    and keeps the longest evidence_quote."""
    gi = _new_instance(tmp_path)
    a = _make_node("A")
    b = _make_node("B")
    gi.storage.add_node(a)
    gi.storage.add_node(b)

    e1 = _make_edge(a.id, b.id)
    e1.text_unit_ids = ["c1"]
    e1.weight = 1.0
    e1.evidence_quote = "short"
    gi.storage._g.add_edge(a.id, b.id, data=e1)

    e2 = _make_edge(a.id, b.id)
    e2.text_unit_ids = ["c2", "c3"]
    e2.weight = 1.0
    e2.evidence_quote = "this is a longer evidence sentence"
    gi.storage._g.add_edge(a.id, b.id, data=e2)

    e3 = _make_edge(a.id, b.id)
    e3.text_unit_ids = ["c1", "c4"]
    e3.weight = 1.0
    e3.evidence_quote = ""
    gi.storage._g.add_edge(a.id, b.id, data=e3)

    stats = gi.storage.dedupe_edges()
    assert stats == {"groups_merged": 1, "edges_removed": 2}
    remaining = list(gi.storage.edges())
    assert len(remaining) == 1
    survivor = remaining[0]
    assert sorted(survivor.text_unit_ids) == ["c1", "c2", "c3", "c4"]
    assert survivor.weight == 3.0
    assert survivor.evidence_quote == "this is a longer evidence sentence"


def test_dedupe_edges_keeps_distinct_types(tmp_path: Path) -> None:
    """Same endpoints but different types are NOT duplicates."""
    gi = _new_instance(tmp_path)
    a = _make_node("A")
    b = _make_node("B")
    gi.storage.add_node(a)
    gi.storage.add_node(b)
    e1 = _make_edge(a.id, b.id)
    e1.type = "RELATED_TO"
    e2 = _make_edge(a.id, b.id)
    e2.type = "LOCATED_IN"
    gi.storage._g.add_edge(a.id, b.id, data=e1)
    gi.storage._g.add_edge(a.id, b.id, data=e2)
    stats = gi.storage.dedupe_edges()
    assert stats["edges_removed"] == 0
    assert sum(1 for _ in gi.storage.edges()) == 2


def test_dedupe_edges_predicate_scopes_to_run(tmp_path: Path) -> None:
    """The ingest path passes a predicate that scopes dedup to the
    current run_id only — cross-document duplicates are independent
    confirmations and must survive."""
    gi = _new_instance(tmp_path)
    a = _make_node("A")
    b = _make_node("B")
    gi.storage.add_node(a)
    gi.storage.add_node(b)

    # Two edges from run-1, one edge from run-2 (different doc).
    e1 = _make_edge(a.id, b.id)
    e1.provenance.extraction_run_id = "run-1"
    e2 = _make_edge(a.id, b.id)
    e2.provenance.extraction_run_id = "run-1"
    e3 = _make_edge(a.id, b.id)
    e3.provenance.extraction_run_id = "run-2"
    gi.storage._g.add_edge(a.id, b.id, data=e1)
    gi.storage._g.add_edge(a.id, b.id, data=e2)
    gi.storage._g.add_edge(a.id, b.id, data=e3)

    stats = gi.storage.dedupe_edges(
        predicate=lambda e: e.provenance.extraction_run_id == "run-1",
    )
    assert stats == {"groups_merged": 1, "edges_removed": 1}
    # run-2's edge is still distinct because the predicate filtered it
    # out of the dedup pool, leaving 2 edges total: the run-1 survivor
    # and the run-2 edge sitting alongside it.
    assert sum(1 for _ in gi.storage.edges()) == 2


def test_dedupe_edges_idempotent(tmp_path: Path) -> None:
    gi = _new_instance(tmp_path)
    a = _make_node("A")
    b = _make_node("B")
    gi.storage.add_node(a)
    gi.storage.add_node(b)
    gi.storage._g.add_edge(a.id, b.id, data=_make_edge(a.id, b.id))
    gi.storage._g.add_edge(a.id, b.id, data=_make_edge(a.id, b.id))
    gi.storage.dedupe_edges()
    second = gi.storage.dedupe_edges()
    assert second == {"groups_merged": 0, "edges_removed": 0}


def test_maybe_log_entity_proposal() -> None:
    from src.modules.m1_ingest import _maybe_log_entity_proposal, parse_ontology
    ont = (Path(__file__).resolve().parent.parent / "ontology.md").read_text(encoding="utf-8")
    ents, _, _ = parse_ontology(ont)

    events = []
    # In ontology — no log
    _maybe_log_entity_proposal("Person", "Elon", ents, events.append,
                               raw_doc_id="d.pdf", run_id="r1", page_num=1)
    # Out of ontology — log
    _maybe_log_entity_proposal("Government_Agency", "EPA", ents, events.append,
                               raw_doc_id="d.pdf", run_id="r1", page_num=1)
    assert len(events) == 1
    assert events[0]["subkind"] == "entity_type_proposed"
    assert events[0]["proposed_type"] == "Government_Agency"
