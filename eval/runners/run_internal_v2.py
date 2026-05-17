"""Run SUT-E: internal Q&A pipeline (deterministic GraphRAG) over the gold testset.

Calls ``src.modules.m2_qa.qa_trace()`` directly — no eval-local copy of the
pipeline. Trace dict carries selected_chunk_ids / sub_questions / max_seed_score
etc. for the JSONL record; ``qa()`` is the thin wrapper that returns only the
final answer for chat callers.

Output: eval/reports/raw_runs/internal_v2.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.runners._common import (
    chunk_to_evidence_page,
    load_gold,
    open_run_writer,
    timed,
    warmup,
    write_run_record,
)
from src.graph.instance import GraphInstance
from src.llm.local_client import LocalClient
from src.modules.m2_qa import qa_trace


def _build_instance() -> GraphInstance:
    return GraphInstance(
        name="production",
        storage_path=ROOT / "data" / "production",
        ontology_path=ROOT / "ontology.md",
    )


def main():
    instance = _build_instance()
    client = LocalClient()

    print("warming up…")
    warmup(instance)
    try:
        qa_trace(instance, "What is sugar kelp?", client=client)
    except Exception as exc:
        print(f"  warm-up failed (continuing): {exc}")

    gold = load_gold()
    fh, path = open_run_writer("internal_v2")
    print(f"writing {path}")
    print(f"running {len(gold)} questions through internal_v2 SUT…")

    for i, q in enumerate(gold, 1):
        question_id = q["question_id"]
        category = q["category"]
        question = q["question"]

        with timed() as tt:
            try:
                r = qa_trace(instance, question, client=client)
                err = None
            except Exception as exc:
                r = {"answer": f"(runner error: {exc})", "error": str(exc)}
                err = str(exc)

        # Map selected_chunk_ids → retrieved_evidence (page-level for scoring)
        retrieved_evidence = []
        for cid in r.get("selected_chunk_ids", []) or []:
            ep = chunk_to_evidence_page(instance, cid)
            if ep:
                retrieved_evidence.append({"chunk_id": cid, **ep})

        rec = {
            "question_id": question_id,
            "category": category,
            "system": "internal_v2",
            "question": question,
            "answer": r.get("answer", ""),
            "retrieved_chunk_ids": r.get("selected_chunk_ids", []) or [],
            "retrieved_evidence": retrieved_evidence,
            "retrieved_doc_titles": r.get("selected_doc_titles", []) or [],
            "latency_ms": tt["ms"],
            "llm_calls": r.get("llm_calls", 0),
            "sub_questions": r.get("sub_questions", []),
            "max_seed_score": r.get("max_seed_score"),
            "n_visited_nodes": r.get("n_visited"),
            "n_chunks_collected": r.get("n_chunks_collected"),
            "n_edges_collected": r.get("n_edges_collected"),
            "oos_prefilter_triggered": r.get("oos_prefilter_triggered", False),
            "oos_postfilter_triggered": r.get("oos_postfilter_triggered", False),
            "raw_answer_before_postfilter": r.get("raw_answer_before_postfilter"),
            "timings_ms": r.get("timings_ms", {}),
            "error": err,
        }
        write_run_record(fh, rec)
        if i % 5 == 0 or i == len(gold):
            print(f"  [{i}/{len(gold)}] {question_id} {category} {tt['ms']}ms calls={rec['llm_calls']} subs={len(rec['sub_questions'])}")

    fh.close()
    print("done.")


if __name__ == "__main__":
    main()
