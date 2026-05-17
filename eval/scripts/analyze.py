"""Aggregate raw_runs/*.jsonl into a scored CSV + a starter summary.

Computes the mechanical metrics (page-level recall/precision, latency,
LLM-call cost). Answer-correctness / faithfulness / refusal-correctness
are LEFT EMPTY here — those are filled in by Claude reading the JSONL
files in conversation. The CSV gives the framework; the analyst (me)
attaches qualitative scores.

Outputs:
  eval/reports/scored.csv          long table: question × system × metric
  eval/reports/aggregate.json      summary stats by category × system
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GOLD_PATH = ROOT / "eval" / "testset" / "gold.jsonl"
RUNS_DIR = ROOT / "eval" / "reports" / "raw_runs"
OUT_CSV = ROOT / "eval" / "reports" / "scored.csv"
OUT_JSON = ROOT / "eval" / "reports" / "aggregate.json"

_PAGES_SUFFIX_RE = re.compile(r"#pages_(\d+)_(\d+)$")


def normalize_doc_page(raw_doc_id: str, page_num: int | None) -> tuple[str, int | None]:
    """Project chunks use super-chunk-local page_num + a #pages_X_Y suffix on
    raw_doc_id. Convert to (basename, global_page) so it matches baseline's
    coordinate system (basename + global page from pdfplumber).
    """
    m = _PAGES_SUFFIX_RE.search(raw_doc_id)
    if m and page_num is not None:
        start = int(m.group(1))
        global_page = start + page_num - 1
        return _PAGES_SUFFIX_RE.sub("", raw_doc_id), global_page
    return raw_doc_id, page_num


def evidence_to_pages(rec: dict) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for e in rec.get("retrieved_evidence", []) or []:
        d, p = normalize_doc_page(e.get("raw_doc_id", ""), e.get("page_num"))
        if p is not None:
            out.add((d, p))
    return out


def gold_pages(gold: dict) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for blk in gold.get("gold_evidence_pages", []) or []:
        rid = blk.get("raw_doc_id", "")
        for p in blk.get("pages", []):
            d, gp = normalize_doc_page(rid, p)
            if gp is not None:
                out.add((d, gp))
    return out


def load_runs() -> dict[str, list[dict]]:
    runs = {}
    for path in RUNS_DIR.glob("*.jsonl"):
        sys_name = path.stem
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        runs[sys_name] = records
    return runs


def load_gold() -> dict[str, dict]:
    out = {}
    with GOLD_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["question_id"]] = d
    return out


def main():
    gold = load_gold()
    runs = load_runs()

    rows = []
    for sys_name, records in runs.items():
        for rec in records:
            qid = rec["question_id"]
            g = gold.get(qid)
            if not g:
                continue
            cat = g["category"]
            gp = gold_pages(g)
            rp = evidence_to_pages(rec)

            # page-level recall (any-of-gold): is at least one gold page in retrieved?
            if cat == "oos":
                recall_any = None
                recall_frac = None
                precision = None
            elif not gp:
                recall_any = None
                recall_frac = None
                precision = None
            else:
                inter = gp & rp
                recall_any = 1 if inter else 0
                recall_frac = round(len(inter) / len(gp), 4) if gp else None
                precision = round(len(inter) / len(rp), 4) if rp else 0.0

            # F1
            if (recall_frac is not None and precision is not None
                    and (recall_frac + precision) > 0):
                f1 = round(
                    2 * recall_frac * precision / (recall_frac + precision), 4
                )
            else:
                f1 = None

            # MRR — rank of first gold-matching retrieved page
            mrr = None
            if cat != "oos" and gp:
                rank = 0
                first = None
                for e in (rec.get("retrieved_evidence", []) or []):
                    d, p = normalize_doc_page(e.get("raw_doc_id", ""), e.get("page_num"))
                    if p is None:
                        continue
                    rank += 1
                    if (d, p) in gp and first is None:
                        first = rank
                        break
                mrr = round(1.0 / first, 4) if first else 0.0

            rows.append({
                "question_id": qid,
                "category": cat,
                "system": sys_name,
                "n_gold_pages": len(gp),
                "n_retrieved_pages": len(rp),
                "n_intersection": len(gp & rp) if gp else 0,
                "recall_at_k": recall_any,       # binary recall@k (=any gold retrieved)
                "recall_frac": recall_frac,       # |gold ∩ retrieved| / |gold|
                "precision_at_k": precision,      # |gold ∩ retrieved| / |retrieved|
                "f1_at_k": f1,
                "mrr": mrr,
                "latency_ms": rec.get("latency_ms"),
                "llm_calls": rec.get("llm_calls"),
                "answer_correctness": "",
                "faithfulness": "",
                "refusal_correct": "",
                "notes": "",
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {OUT_CSV}: {len(rows)} rows")

    # aggregate
    agg = defaultdict(lambda: defaultdict(list))  # (cat, system) -> metric -> values
    for r in rows:
        key = (r["category"], r["system"])
        for m in ("recall_at_k", "recall_frac", "precision_at_k", "f1_at_k", "mrr",
                  "latency_ms", "llm_calls"):
            v = r.get(m)
            if v is not None:
                agg[key][m].append(v)

    summary = {}
    for (cat, sys_name), metrics in agg.items():
        d = {}
        for m, vs in metrics.items():
            if not vs:
                continue
            try:
                d[f"{m}_mean"] = round(sum(vs) / len(vs), 4)
                d[f"{m}_n"] = len(vs)
            except Exception:
                pass
        # Also median latency
        if "latency_ms" in metrics and metrics["latency_ms"]:
            xs = sorted(metrics["latency_ms"])
            d["latency_ms_p50"] = xs[len(xs) // 2]
            d["latency_ms_p95"] = xs[int(len(xs) * 0.95)] if len(xs) >= 5 else xs[-1]
        summary.setdefault(cat, {})[sys_name] = d

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"wrote {OUT_JSON}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
