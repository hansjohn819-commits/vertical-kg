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
        # Sync lock: while a sleep pass is running, the M2 agent refuses new
        # requests (§14 wants sync semantics — pass runs matter more than chat
        # availability). Implementation: simple bool, no threading needed since
        # Phase 6 runs sleep pass synchronously in the agent's own process.
        self.sleep_pass_running: bool = False

    # --- Module delegates (filled in by later phases) ---

    def ingest(self, raw_text: str, raw_doc_id: str):
        """M1: raw → initial graph."""
        from src.modules.m1_ingest import ingest_raw_text
        return ingest_raw_text(self.storage, raw_text, raw_doc_id)

    def qa(self, question: str) -> str:
        """M2: agent-based Q&A."""
        from src.modules.m2_qa_agent import GraphAgent
        return GraphAgent(self).call(question)

    def integrate(
        self,
        statement: str,
        user_id: str,
        conversation_id: str,
        turn_id: str,
    ):
        """M3: online write from chat."""
        from src.modules.m3_integrate import UserContext, integrate_user_statement
        ctx = UserContext(
            user_id=user_id, conversation_id=conversation_id, turn_id=turn_id
        )
        return integrate_user_statement(self.storage, statement, ctx)

    def sleep_pass(self) -> dict:
        """M4: periodic maintenance (4c→4b→4a→4d)."""
        from src.modules.m4_sleep_pass.runner import run_sleep_pass
        return run_sleep_pass(self)

    # --- Utilities ---

    def stats(self) -> dict:
        return {"instance": self.name, **self.storage.stats()}

    def save(self) -> None:
        self.storage.save()
