"""GraphInstance: one graph world (production or experiment).

Same class, two instances — see guide §2.2 / §9.3. Module hooks (ingest, qa,
sleep_pass) delegate to M1–M4 implementations.

Carries both `storage` (NetworkX truth) and `vector_store` (FAISS, §16.10).
Boot-time invariant: vector_store node count == storage active-node count.
On mismatch the instance auto-rebuilds the FAISS index from storage —
crash recovery + first-time bootstrap share the same path.
"""

from pathlib import Path

from .bm25_store import BM25Store
from .retrieval import EMBEDDING_DIM, encode_node
from .storage import GraphStorage
from .text_units import TextUnitStore
from .vector_store import VectorStore


class GraphInstance:
    def __init__(
        self,
        name: str,
        storage_path: str | Path,
        ontology_path: str | Path,
    ):
        self.name = name
        sp = Path(storage_path)
        self.storage = GraphStorage(sp / "graph.pkl")
        self.storage.load()
        self.vector_store = VectorStore(
            index_path=sp / "embeddings.faiss",
            ids_path=sp / "embedding_ids.json",
            dim=EMBEDDING_DIM,
        )
        self._ensure_vector_store_consistent()
        # §16.20 lexical companion to FAISS for the external retrieval path.
        # Boot guard rebuilds from storage if file missing / size mismatch —
        # mirrors the FAISS pattern, no manual migration script needed.
        self.bm25_store = BM25Store(sp / "bm25.pkl")
        self._ensure_bm25_consistent()
        # §16.17 source-text persistence layer.  Loaded after storage so the
        # consistency check below has the authoritative node/edge set to
        # cross-reference against.
        self.text_units = TextUnitStore(sp / "text_units.json")
        self.text_units.load()
        self._ensure_text_units_consistent()
        self.ontology_path = Path(ontology_path)
        # Sync lock: while a sleep pass is running, the M2 agent refuses new
        # requests (§14 wants sync semantics — pass runs matter more than chat
        # availability). Implementation: simple bool, no threading needed since
        # Phase 6 runs sleep pass synchronously in the agent's own process.
        self.sleep_pass_running: bool = False

    def _active_node_count(self) -> int:
        return sum(1 for n in self.storage.nodes() if n.merged_into is None)

    def _ensure_vector_store_consistent(self) -> None:
        """Boot-time guard. Loads index from disk; if files are missing,
        corrupt, or count-mismatched against storage, rebuilds in place.
        Mismatch is the only signal we have for crash-mid-write — the cost
        of a full rebuild at startup (~10s for 200 nodes) is much cheaper
        than running queries against a stale index.
        """
        loaded = self.vector_store.load()
        if loaded and self.vector_store.size == self._active_node_count():
            return
        # Either no/broken index, or storage and vector got out of sync.
        # Rebuild from authoritative storage. Ghosts are excluded.
        pairs = [
            (n.id, encode_node(n))
            for n in self.storage.nodes()
            if n.merged_into is None
        ]
        self.vector_store.rebuild_from(pairs)
        self.vector_store.save()

    def _ensure_bm25_consistent(self) -> None:
        """Boot-time guard mirroring _ensure_vector_store_consistent. BM25
        is always fully refit (no incremental API), so the check is just
        "did the file load AND does row count match active nodes". On
        mismatch, full refit + save. First boot after deploy (file
        missing) takes the same path — no manual migration script."""
        loaded = self.bm25_store.load()
        if loaded and self.bm25_store.size == self._active_node_count():
            return
        self.bm25_store.fit_from(self.storage)
        self.bm25_store.save()

    def _ensure_text_units_consistent(self) -> None:
        """Boot-time guard for the §16.17 source-text layer.

        Unlike the FAISS index (which can be rebuilt deterministically from
        storage), text_units.json is the canonical store of raw page text —
        if it's lost we can't reconstitute it without re-ingesting the
        source PDFs.  So the recovery story is "best-effort align, log
        anomalies, never raise" rather than "wipe and rebuild".

        Only one failure mode is handled here: a node/edge references a
        chunk_id that no longer exists in text_units.  Likely cause: a
        manual edit of text_units.json went wrong, or the file was
        replaced with an older copy.  Action: scrub the dangling ids
        from node/edge fields so query paths don't blow up.

        We do NOT delete chunks that no node/edge currently references.
        That sounds like "orphan cleanup" but is actually dangerous: when
        ingest crashes mid-PASS-1, the chunks for the pages PASS 1 hadn't
        reached yet have no incoming references — and deleting them
        destroys the only on-disk copy of that page text, blocking any
        subsequent retry from continuing where the failed run left off.
        Letting unreferenced chunks sit costs near-zero (~3 KB each) and
        keeps the door open for resume / partial retry.
        """
        expected: set[str] = set()
        for n in self.storage.nodes():
            expected.update(n.text_unit_ids or [])
        for e in self.storage.edges():
            expected.update(e.text_unit_ids or [])

        actual = self.text_units.keys()
        missing = expected - actual
        if not missing:
            return

        for n in self.storage.nodes():
            if n.text_unit_ids and any(t in missing for t in n.text_unit_ids):
                n.text_unit_ids = [t for t in n.text_unit_ids if t not in missing]
        for e in self.storage.edges():
            if e.text_unit_ids and any(t in missing for t in e.text_unit_ids):
                e.text_unit_ids = [t for t in e.text_unit_ids if t not in missing]
        self.storage.save()

    # --- Module delegates (filled in by later phases) ---

    def ingest(self, pages: list[str], raw_doc_id: str):
        """M1: raw → initial graph (§16.17 per-page two-pass pipeline).

        ``pages`` is an ordered list of page-text strings. Single-element
        list is the right shape for non-paginated sources (.txt / .md
        / single-page PDF). Multi-element triggers the full PASS 2 +
        intra-doc fuse logic in :func:`m1_ingest.ingest_document`.
        """
        from src.modules.m1_ingest import ingest_document
        return ingest_document(
            storage=self.storage,
            vector_store=self.vector_store,
            text_units=self.text_units,
            ontology_path=self.ontology_path,
            raw_doc_id=raw_doc_id,
            pages=pages,
        )

    def qa(self, question: str, history: list[dict] | None = None) -> str:
        """M2 Q&A: deterministic GraphRAG-style pipeline (no agent loop).
        See src.modules.m2_qa for the full design + parameters.

        For chat command parsing (``/ingest <file>``, ``/sleep``), use
        :class:`src.modules.m2_qa_agent.GraphAgent` which routes commands to
        the appropriate tool and delegates plain questions to this method.
        """
        from src.modules.m2_qa import qa
        return qa(self, question, history=history)

    def show_provenance(self, node_or_edge_id: str) -> dict:
        """Resolve a node OR edge ID to its provenance + source text chunks.

        Returns one of:
          {"kind": "node", "id", "label", "type", "provenance", "sources"?}
          {"kind": "edge", "id", "type", "source": {...}, "target": {...},
           "provenance", "evidence_quote"?, "sources"?}
          {"kind": "not_found", "id": ...}

        `sources` (if present) is a list of {"raw_doc_id", "page_num",
        "text"} dicts resolved from text_unit_ids.

        Python API only — not chat-accessible after the 2026-05-15 router
        refactor. Streamlit / admin UIs / scripts can call this directly.
        """
        def _resolve(chunk_ids):
            if not chunk_ids:
                return []
            out = []
            for cd in self.text_units.get_many(chunk_ids):
                out.append({
                    "raw_doc_id": cd.get("raw_doc_id", ""),
                    "page_num": cd.get("page_num"),
                    "text": cd.get("text", ""),
                })
            return out

        n = self.storage.get_node(node_or_edge_id)
        if n is not None:
            payload = {
                "kind": "node",
                "id": n.id,
                "label": n.label,
                "type": n.type,
                "provenance": n.provenance.model_dump(),
            }
            sources = _resolve(n.text_unit_ids)
            if sources:
                payload["sources"] = sources
            return payload

        e = self.storage.get_edge(node_or_edge_id)
        if e is not None:
            src = self.storage.get_node(e.source_id)
            tgt = self.storage.get_node(e.target_id)
            payload = {
                "kind": "edge",
                "id": e.id,
                "type": e.type,
                "source": {"id": e.source_id,
                           "label": src.label if src else None},
                "target": {"id": e.target_id,
                           "label": tgt.label if tgt else None},
                "provenance": e.provenance.model_dump(),
            }
            if e.evidence_quote:
                payload["evidence_quote"] = e.evidence_quote
            sources = _resolve(e.text_unit_ids)
            if sources:
                payload["sources"] = sources
            return payload

        return {"kind": "not_found", "id": node_or_edge_id}

    def sleep_pass(self) -> dict:
        """M4: periodic maintenance (4c→4b→4a→4d)."""
        from src.modules.m4_sleep_pass.runner import run_sleep_pass
        return run_sleep_pass(self)

    # --- Utilities ---

    def stats(self) -> dict:
        return {"instance": self.name, **self.storage.stats()}

    def save(self) -> None:
        self.storage.save()
        self.vector_store.save()
        self.text_units.save()
        # §16.20: BM25 has no incremental API → full refit at every save().
        # Cost is ~100ms on 2.4k nodes; save() is called only at M1 ingest
        # end / M4 sleep pass end / super-chunk boundaries (a few times per
        # ingest, once per sleep pass), so total daily refit cost is well
        # under 1 second.
        self.bm25_store.fit_from(self.storage)
        self.bm25_store.save()
        # §16.6 task 3: refresh the cached node layout so the next audit
        # view load can render with `layout: 'preset'` (zero browser
        # compute) instead of running fcose in JS. Recompute on save
        # covers every topology change — M1 ingest, M4 sleep pass,
        # normalize_edges, dedupe_edges all funnel through here.
        try:
            from src.dashboard.layout_cache import ensure_layout
            ensure_layout(self.storage.path.parent, self.storage)
        except Exception:
            # Layout is a UI optimization, not authoritative state.
            # Failure here (e.g., networkx missing, transient I/O error)
            # must not block a successful graph save.
            pass
