"""Run SUT-D: external fast-query path over the gold testset.

Calls ``src.modules.m2_qa_agent.fast_query_trace()`` directly. The trace dict
exposes selected_chunk_ids / seeds / max_seed_score / timings for the JSONL
record. ``fast_query()`` is the thin wrapper that returns only the answer
string for external chat callers.

Output: eval/reports/raw_runs/external_v2.jsonl
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
from src.modules.m2_qa_agent import fast_query_trace


def _build_instance() -> GraphInstance:
    return GraphInstance(
        name="production",
        storage_path=ROOT / "data" / "production",
        ontology_path=ROOT / "ontology.md",
    )


def main():
    instance = _build_instance()

    print("warming up…")
    warmup(instance)
    try:
        fast_query_trace(instance, "What is sugar kelp?")
    except Exception as exc:
        print(f"  warm-up failed (continuing): {exc}")

    gold = load_gold()
    fh, path = open_run_writer("external_v2")
    print(f"writing {path}")
    print(f"running {len(gold)} questions through external_v2 SUT…")

    for i, q in enumerate(gold, 1):
        question_id = q["question_id"]
        category = q["category"]
        question = q["question"]

        with timed() as tt:
            try:
                r = fast_query_trace(instance, question)
                err = None
            except Exception as exc:
                r = {"answer": f"(runner error: {exc})", "error": str(exc)}
                err = str(exc)

        retrieved_evidence = []
        for cid in r.get("selected_chunk_ids", []) or []:
            ep = chunk_to_evidence_page(instance, cid)
            if ep:
                retrieved_evidence.append({"chunk_id": cid, **ep})

        rec = {
            "question_id": question_id,
            "category": category,
            "system": "external_v2",
            "question": question,
            "answer": r.get("answer", ""),
            "retrieved_chunk_ids": r.get("selected_chunk_ids", []) or [],
            "retrieved_evidence": retrieved_evidence,
            "retrieved_doc_titles": r.get("selected_doc_titles", []) or [],
            "seeds": r.get("seeds", []),
            "max_seed_score": r.get("max_seed_score"),
            "seed_query": r.get("seed_query"),
            "latency_ms": tt["ms"],
            "llm_calls": r.get("llm_calls", 0),
            "raw_answer_before_postfilter": r.get("raw_answer_before_postfilter"),
            "timings_ms": r.get("timings_ms", {}),
            "error": err,
        }
        write_run_record(fh, rec)
        if i % 5 == 0 or i == len(gold):
            print(f"  [{i}/{len(gold)}] {question_id} {category} {tt['ms']}ms calls={rec['llm_calls']} seeds={len(rec['seeds'])}")

    fh.close()
    print("done.")


if __name__ == "__main__":
    main()
