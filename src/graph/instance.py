"""GraphInstance: one graph world (production or experiment).

Same class, two instances — see guide §2.2 / §9.3. Module hooks (ingest, qa,
integrate, sleep_pass) are declared here and delegate to M1–M4 implementations
as those phases land.
"""

from pathlib import Path

from .storage import GraphStorage


class GraphInstance:
    def __init__(
        self,
        name: str,
        storage_path: str | Path,
        ontology_path: str | Path,
    ):
        self.name = name
        self.storage = GraphStorage(Path(storage_path) / "graph.pkl")
        self.storage.load()
        self.ontology_path = Path(ontology_path)

    # --- Module delegates (filled in by later phases) ---

    def ingest(self, raw_text: str, raw_doc_id: str) -> None:
        """M1: raw → initial graph. Implemented in Phase 4."""
        raise NotImplementedError("M1 ingest — Phase 4")

    def qa(self, question: str) -> str:
        """M2: agent-based Q&A. Implemented in Phase 5."""
        raise NotImplementedError("M2 qa — Phase 5")

    def integrate(
        self,
        fact_text: str,
        user_id: str,
        conversation_id: str,
        turn_id: str,
    ) -> None:
        """M3: online write from chat. Implemented in Phase 4."""
        raise NotImplementedError("M3 integrate — Phase 4")

    def sleep_pass(self) -> dict:
        """M4: periodic maintenance (4c→4b→4a→4d). Implemented in Phase 6."""
        raise NotImplementedError("M4 sleep_pass — Phase 6")

    # --- Utilities ---

    def stats(self) -> dict:
        return {"instance": self.name, **self.storage.stats()}

    def save(self) -> None:
        self.storage.save()
