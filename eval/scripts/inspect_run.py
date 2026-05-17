"""Quick inspector — load gold + a runner output side by side for analysis.

Usage: python -m eval.scripts.inspect_run [--system internal|external|baseline]
                                          [--category single_hop|multi_hop|aggregation|oos]
                                          [--qid q001 …]

For each requested record, print:
  - question + gold facts + gold pages
  - system answer + retrieved (doc, page) tuples
  - did retrieval hit any gold page?

Used by Claude in conversation to score answer correctness manually.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from eval.scripts.analyze import (  # noqa: E402
    evidence_to_pages, gold_pages, load_gold,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default=None)
    ap.add_argument("--category", default=None)
    ap.add_argument("--qid", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--full-answer", action="store_true",
                    help="print full answer text (default truncates at 600 chars)")
    args = ap.parse_args()

    gold = load_gold()

    runs_dir = ROOT / "eval" / "reports" / "raw_runs"
    sys_files = (
        [runs_dir / f"{args.system}.jsonl"] if args.system
        else sorted(runs_dir.glob("*.jsonl"))
    )

    n = 0
    for path in sys_files:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                qid = rec["question_id"]
                if args.qid and qid not in args.qid:
                    continue
                if args.category and rec["category"] != args.category:
                    continue
                g = gold.get(qid)
                if not g:
                    continue
                gp = gold_pages(g)
                rp = evidence_to_pages(rec)
                hit = bool(gp & rp)

                print("=" * 100)
                print(f"[{rec['system']}] {qid} ({rec['category']}) latency={rec.get('latency_ms')}ms calls={rec.get('llm_calls')}")
                print(f"Q: {rec['question']}")
                print(f"GOLD facts: {g['gold_facts']}")
                if rec["category"] != "oos":
                    print(f"GOLD pages ({len(gp)}): {sorted(gp)[:8]}{'...' if len(gp)>8 else ''}")
                    print(f"RETRIEVED pages ({len(rp)}): {sorted(rp)[:8]}{'...' if len(rp)>8 else ''}")
                    print(f"HIT: {hit}  intersection={sorted(gp & rp)[:5]}")
                    print(f"expected: {g['expected_behavior']}")
                else:
                    print(f"OOS — expected: refuse")
                ans = rec.get("answer", "")
                if not args.full_answer and len(ans) > 600:
                    ans = ans[:600] + "…"
                print(f"ANSWER: {ans}")
                print()
                n += 1
                if args.limit and n >= args.limit:
                    return


if __name__ == "__main__":
    main()
