"""GraphInstance: one graph world (production or experiment).

Same class, two instances — see guide §2.2 / §9.3. Module hooks (ingest, qa,
sleep_pass) delegate to M1–M4 implementations.

Carries both `storage` (NetworkX truth) and `vector_store` (FAISS, §16.10).
Boot-time invariant: vector_store node count == storage active-node count.
On mismatch the instance auto-rebuilds the FAISS index from storage —
crash recovery + first-time bootstrap share the same path.
"""

from pathlib import Path

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

    def qa(self, question: str) -> str:
        """M2: agent-based Q&A."""
        from src.modules.m2_qa_agent import GraphAgent
        return GraphAgent(self).call(question)

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
