"""Run SUT-A: internal GraphAgent.call() over the gold testset.

For each question:
  1. Pre-compute selected chunk_ids by replicating the project's chunk
     selection logic (same FAISS + rerank + budget).
  2. Run agent.call(question, history=[]) to get the final answer (the
     agent loop will internally redo the same retrieval — deterministic).
  3. Record latency, answer text, retrieved chunk_ids → (doc, page) tuples,
     llm call count (approximate: counted via a request hook).

Output: eval/reports/raw_runs/internal.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.runners._common import (
    chunk_to_evidence_page,
    get_selected_chunks,
    load_gold,
    open_run_writer,
    timed,
    warmup,
    write_run_record,
)
from src.graph.instance import GraphInstance
from src.modules.m2_qa_agent import GraphAgent


def _build_instance() -> GraphInstance:
    return GraphInstance(
        name="production",
        storage_path=ROOT / "data" / "production",
        ontology_path=ROOT / "ontology.md",
    )


def _patch_llm_counter():
    """Wrap LocalClient.chat to count calls. Returns a getter that resets."""
    from src.llm import local_client
    original = local_client.LocalClient.chat
    state = {"count": 0}

    def counted(self, *args, **kwargs):
        state["count"] += 1
        return original(self, *args, **kwargs)

    local_client.LocalClient.chat = counted

    def reset_and_get():
        n = state["count"]
        state["count"] = 0
        return n
    return reset_and_get


def main():
    instance = _build_instance()
    agent = GraphAgent(instance)

    print("warming up…")
    warmup(instance)
    # One throwaway agent call to warm tool routing + LLM caches.
    try:
        agent.call("What is sugar kelp?", history=[])
    except Exception as exc:
        print(f"  warm-up call failed (continuing): {exc}")

    counter = _patch_llm_counter()

    gold = load_gold()
    fh, path = open_run_writer("internal")
    print(f"writing {path}")
    print(f"running {len(gold)} questions through internal SUT…")

    for i, q in enumerate(gold, 1):
        question_id = q["question_id"]
        category = q["category"]
        question = q["question"]

        try:
            selected, seeds_payload = get_selected_chunks(
                instance, question, use_display_title=False,
            )
        except Exception as exc:
            selected, seeds_payload = [], []
            print(f"  [{question_id}] selection failed: {exc}")

        evidence = []
        for cid in selected:
            ep = chunk_to_evidence_page(instance, cid)
            if ep:
                evidence.append({"chunk_id": cid, **ep})

        counter()  # reset
        with timed() as tt:
            try:
                answer = agent.call(question, history=[])
            except Exception as exc:
                answer = f"(runner error: {exc})"
        llm_calls = counter()

        rec = {
            "question_id": question_id,
            "category": category,
            "system": "internal",
            "question": question,
            "answer": answer,
            "retrieved_chunk_ids": selected,
            "retrieved_evidence": evidence,
            "seeds": seeds_payload,
            "latency_ms": tt["ms"],
            "llm_calls": llm_calls,
        }
        write_run_record(fh, rec)
        if i % 5 == 0 or i == len(gold):
            print(f"  [{i}/{len(gold)}] {question_id} {category} {tt['ms']}ms calls={llm_calls}")

    fh.close()
    print("done.")


if __name__ == "__main__":
    main()
