"""Run SUT-C: independent RAG+BM25 baseline over the gold testset.

Loads eval/data/baseline_index/ artifacts, runs hybrid RRF retrieve, feeds
top-10 chunks to the baseline composer.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.baseline.bm25_store import BaselineBM25Store
from eval.baseline.composer import compose
from eval.baseline.embed_store import BaselineDenseStore, load_chunks
from eval.baseline.retriever import HybridRetriever
from eval.runners._common import (
    baseline_chunk_to_evidence_pages,
    load_gold,
    open_run_writer,
    timed,
    write_run_record,
)
from src.llm.local_client import LocalClient


def _patch_llm_counter():
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
    idx_dir = ROOT / "eval" / "data" / "baseline_index"
    chunks = load_chunks(idx_dir)
    chunks_by_id = {c["id"]: c for c in chunks}
    dense = BaselineDenseStore()
    dense.load(idx_dir)
    bm25 = BaselineBM25Store()
    bm25.load(idx_dir)
    retr = HybridRetriever(dense, bm25, chunks_by_id)
    client = LocalClient()

    print("warming up…")
    # Warm: embed one query, run a tiny LLM call.
    _ = retr.retrieve("What is sugar kelp?")
    try:
        compose("warmup", _, client=client)
    except Exception as exc:
        print(f"  warm-up call failed (continuing): {exc}")

    counter = _patch_llm_counter()
    gold = load_gold()
    fh, path = open_run_writer("baseline")
    print(f"writing {path}")
    print(f"running {len(gold)} questions through baseline SUT (n_chunks={len(chunks)})…")

    for i, q in enumerate(gold, 1):
        question_id = q["question_id"]
        category = q["category"]
        question = q["question"]

        with timed() as t_retrieve:
            try:
                hits = retr.retrieve(question)
            except Exception as exc:
                hits = []
                print(f"  [{question_id}] retrieve failed: {exc}")

        retrieved_chunk_ids = [c["id"] for c in hits]
        retrieved_evidence = []
        for c in hits:
            for ep in baseline_chunk_to_evidence_pages(c):
                retrieved_evidence.append(ep)
        retrieved_doc_titles = sorted({c.get("doc_title", "") for c in hits if c.get("doc_title")})

        counter()
        with timed() as t_compose:
            try:
                answer = compose(question, hits, client=client)
            except Exception as exc:
                answer = f"(runner error: {exc})"
        llm_calls = counter()

        rec = {
            "question_id": question_id,
            "category": category,
            "system": "baseline",
            "question": question,
            "answer": answer,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_evidence": retrieved_evidence,
            "retrieved_doc_titles": retrieved_doc_titles,
            "latency_ms": (t_retrieve["ms"] or 0) + (t_compose["ms"] or 0),
            "latency_retrieve_ms": t_retrieve["ms"],
            "latency_compose_ms": t_compose["ms"],
            "llm_calls": llm_calls,
        }
        write_run_record(fh, rec)
        if i % 5 == 0 or i == len(gold):
            print(f"  [{i}/{len(gold)}] {question_id} {category} {rec['latency_ms']}ms calls={llm_calls}")

    fh.close()
    print("done.")


if __name__ == "__main__":
    main()
