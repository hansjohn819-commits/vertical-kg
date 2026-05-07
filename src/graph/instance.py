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

    # --- Module delegates (filled in by later phases) ---

    def ingest(self, raw_text: str, raw_doc_id: str):
        """M1: raw → initial graph."""
        from src.modules.m1_ingest import ingest_raw_text
        return ingest_raw_text(self.storage, raw_text, raw_doc_id, vector_store=self.vector_store)

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
