"""Export compact per-(question × system) view for relevance judgment.

For each question (excluding OOS) and each system, output:
  - question text
  - gold_facts (1-N short facts the answer should hit)
  - retrieved chunks: [(chunk_id, doc_title, page, text_preview_~300chars)]

Output: eval/reports/relevance_dump.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.graph.text_units import TextUnitStore
from src.graph.retrieval import display_title


PREVIEW_CHARS = 300
SYSTEMS = ["baseline", "external_v2", "internal_v2"]


def main():
    gold: dict[str, dict] = {}
    with (ROOT / "eval" / "testset" / "gold.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            gold[d["question_id"]] = d

    runs: dict[str, dict] = {s: {} for s in SYSTEMS}
    for s in SYSTEMS:
        with (ROOT / "eval" / "reports" / "raw_runs" / f"{s}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                runs[s][d["question_id"]] = d

    # Load project text_units (for internal/external chunk text lookup).
    tu = TextUnitStore(ROOT / "data" / "production" / "text_units.json")
    tu.load()

    # Load baseline chunks (different format — chunk_id from baseline_index pickle).
    import pickle
    with (ROOT / "eval" / "data" / "baseline_index" / "chunks.pkl").open("rb") as f:
        baseline_chunks_list = pickle.load(f)
    baseline_chunks = {c["id"]: c for c in baseline_chunks_list}

    out_path = ROOT / "eval" / "reports" / "relevance_dump.jsonl"
    with out_path.open("w", encoding="utf-8") as outf:
        for qid in sorted(gold):
            g = gold[qid]
            if g["category"] == "oos":
                continue
            rec = {
                "qid": qid,
                "category": g["category"],
                "question": g["question"],
                "gold_facts": g["gold_facts"],
                "gold_doc_titles": g["gold_doc_titles"],
                "systems": {},
            }
            for s in SYSTEMS:
                r = runs[s].get(qid, {})
                chunks_out = []
                for cid in (r.get("retrieved_chunk_ids", []) or []):
                    if s == "baseline":
                        c = baseline_chunks.get(cid)
                        if c is None:
                            continue
                        chunks_out.append({
                            "chunk_id": cid,
                            "doc_title": c.get("doc_title", ""),
                            "page": f"{c.get('page_start')}-{c.get('page_end')}" if c.get("page_start") != c.get("page_end") else str(c.get("page_start")),
                            "text_preview": (c.get("text", "") or "").replace("\n", " ").strip()[:PREVIEW_CHARS],
                        })
                    else:
                        c = tu.get(cid)
                        if c is None:
                            continue
                        chunks_out.append({
                            "chunk_id": cid,
                            "doc_title": display_title(c.get("raw_doc_id", "")),
                            "page": str(c.get("page_num", "")),
                            "text_preview": (c.get("text", "") or "").replace("\n", " ").strip()[:PREVIEW_CHARS],
                        })
                rec["systems"][s] = {
                    "answer": r.get("answer", "")[:300],
                    "retrieved": chunks_out,
                }
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
