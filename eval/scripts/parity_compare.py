"""Compare two raw_runs JSONL files question-by-question.

Used after the §16.22 eval-to-src parity migration: ``before_parity`` runs
were produced by the eval-local v2 lib / re-implemented external pipeline;
``after_parity`` runs call src ``qa_trace`` / ``fast_query_trace`` directly.
This script flags any per-question retrieval or answer divergence so we can
decide whether the migration introduced a regression.

Usage:
  python -m eval.scripts.parity_compare internal_v2
  python -m eval.scripts.parity_compare external_v2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "eval" / "reports" / "raw_runs"


def load(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["question_id"]] = d
    return out


def page_set(rec: dict) -> set[tuple]:
    out = set()
    for e in rec.get("retrieved_evidence", []) or []:
        out.add((e.get("raw_doc_id", ""), e.get("page_num")))
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: parity_compare.py <system_name>")
        sys.exit(2)
    sys_name = sys.argv[1]
    before = load(RUNS_DIR / f"{sys_name}.before_parity.jsonl")
    after = load(RUNS_DIR / f"{sys_name}.jsonl")

    qids = sorted(set(before) | set(after))
    n_total = len(qids)
    n_same_pages = 0
    n_diff_pages = 0
    n_same_answer = 0
    n_diff_answer = 0
    diffs: list[dict] = []

    for qid in qids:
        b = before.get(qid)
        a = after.get(qid)
        if b is None or a is None:
            print(f"  [{qid}] missing in one side (b={b is not None}, a={a is not None})")
            continue
        b_pages = page_set(b)
        a_pages = page_set(a)
        b_ans = (b.get("answer") or "").strip()
        a_ans = (a.get("answer") or "").strip()
        pages_same = (b_pages == a_pages)
        ans_same = (b_ans == a_ans)
        if pages_same:
            n_same_pages += 1
        else:
            n_diff_pages += 1
        if ans_same:
            n_same_answer += 1
        else:
            n_diff_answer += 1
        if not pages_same or not ans_same:
            diffs.append({
                "qid": qid,
                "category": b.get("category"),
                "pages_same": pages_same,
                "answer_same": ans_same,
                "b_pages": sorted(b_pages),
                "a_pages": sorted(a_pages),
                "page_overlap_pct": (
                    round(len(b_pages & a_pages) / max(1, len(b_pages | a_pages)) * 100, 1)
                ),
                "b_answer_excerpt": b_ans[:200],
                "a_answer_excerpt": a_ans[:200],
            })

    print(f"=== {sys_name} parity report ===")
    print(f"questions: {n_total}")
    print(f"retrieved pages identical: {n_same_pages}/{n_total} ({n_same_pages/n_total*100:.1f}%)")
    print(f"answer text identical:     {n_same_answer}/{n_total} ({n_same_answer/n_total*100:.1f}%)")

    print("\n--- per-question divergences ---")
    for d in diffs:
        flags = []
        if not d["pages_same"]:
            flags.append(f"pages(overlap={d['page_overlap_pct']}%)")
        if not d["answer_same"]:
            flags.append("answer")
        print(f"  [{d['qid']}] [{d['category']}] {' '.join(flags)}")


if __name__ == "__main__":
    main()
