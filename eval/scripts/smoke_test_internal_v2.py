"""Run internal_v2 on 7 hand-picked questions covering each category.

Goal: sanity-check the architecture + parameters before committing to a full
61-question run. Prints side-by-side comparison with v1 / external_v2 / gold,
plus traces of decomposer output and traversal frontier per question.

Smoke set:
  q003  single trivial    (Tanzania)        v1=1, v2 should stay 1
  q021  single hard       (EAF-Nansen)      v1=0, v2 should answer
  q034  multi bridge      (Stekoll-Yarish)  v1=0, v2 critical target
  q040  multi             (Mount Desert)    v1=0.5, v2 should be 1
  q046  aggregation       (NA kelp cos)     v1=0 (timeout), v2 baseline = no timeout
  q055  oos               (Python code)     v1=0 (leaked), v2 should refuse
  q061  oos               (Mongolia cap)    v1=0 (leaked), v2 should refuse
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.graph.instance import GraphInstance
from src.llm.local_client import LocalClient
from src.modules.m2_qa import qa_trace


SMOKE_QIDS = ["q003", "q021", "q034", "q040", "q046", "q055", "q061"]


def load_gold():
    gold = {}
    with (ROOT / "eval" / "testset" / "gold.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            gold[d["question_id"]] = d
    return gold


def load_v1():
    runs = {}
    p = ROOT / "eval" / "reports" / "raw_runs" / "internal.jsonl"
    with p.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            runs[d["question_id"]] = d
    return runs


def main():
    gold = load_gold()
    v1 = load_v1()
    instance = GraphInstance(
        name="production",
        storage_path=ROOT / "data" / "production",
        ontology_path=ROOT / "ontology.md",
    )
    client = LocalClient()

    # Warm up: one cheap pass so first real timing isn't penalized
    print("warming up…", flush=True)
    try:
        _ = qa_trace(instance, "What is sugar kelp?", client=client)
    except Exception as exc:
        print(f"  warm-up failed (continuing): {exc}", flush=True)

    out_path = ROOT / "eval" / "reports" / "smoke_internal_v2.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w", encoding="utf-8")

    for qid in SMOKE_QIDS:
        g = gold[qid]
        question = g["question"]
        print("=" * 100, flush=True)
        print(f"[{qid}] [{g['category']}] Q: {question}", flush=True)
        print(f"GOLD: {g['gold_facts'][0] if g['gold_facts'] else '(refuse expected)'}", flush=True)

        v1_ans = v1.get(qid, {}).get("answer", "")[:200]
        print(f"v1: {v1_ans}…", flush=True)

        t0 = time.time()
        try:
            r = qa_trace(instance, question, client=client)
            err = None
        except Exception as exc:
            r = {"answer": f"(runner error: {exc})", "error": str(exc)}
            err = str(exc)
        elapsed = int((time.time() - t0) * 1000)

        print(f"\nv2 [{elapsed}ms, llm_calls={r.get('llm_calls','?')}, max_seed={r.get('max_seed_score',0):.3f}]:", flush=True)
        if r.get("oos_prefilter_triggered"):
            print(f"  ⚠ OOS pre-filter triggered (max_seed_score < threshold)", flush=True)
        if r.get("oos_postfilter_triggered"):
            print(f"  ⚠ OOS post-filter rewrote answer to refusal", flush=True)
            print(f"  Raw answer was: {(r.get('raw_answer_before_postfilter') or '')[:200]}…", flush=True)
        print(f"  sub_questions ({len(r.get('sub_questions', []))}):", flush=True)
        for sq in r.get("sub_questions", []):
            print(f"    - {sq}", flush=True)
        print(f"  visited={r.get('n_visited',0)} chunks_collected={r.get('n_chunks_collected',0)} edges={r.get('n_edges_collected',0)} chunks_in_prompt={len(r.get('selected_chunk_ids', []))}", flush=True)
        ans = r.get("answer", "")
        ans_show = ans[:500] + ("…" if len(ans) > 500 else "")
        print(f"  ANSWER: {ans_show}", flush=True)

        rec = {
            "question_id": qid,
            "category": g["category"],
            "question": question,
            "gold_facts": g["gold_facts"],
            "elapsed_ms": elapsed,
            "v1_answer_excerpt": v1_ans,
            "v2_result": r,
            "error": err,
        }
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        fh.flush()

    fh.close()
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
