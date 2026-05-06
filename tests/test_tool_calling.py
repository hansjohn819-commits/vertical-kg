"""Tool-calling accuracy probe over the §6.2 tool box.

Structure mirrors the external baseline `F:\\Master_CA\\Python Test\\test.py`
(2026-04-20 local-backend probe) but dispatches through our real `GraphAgent`
so we're measuring the deliverable, not a parallel TOOLS array.

Knowledge enters the graph only through `ingest_file` (against documents
the user has placed under data/raw/); there are no chat-side write tools.
The tier-9 "create a node" prompts therefore expect the agent to refuse
or to fall back to a read tool — never an invented write.

Run: python -m tests.test_tool_calling
Results: results/tool_calling.txt (gitignored)
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

from src.graph.instance import GraphInstance
from src.modules.m2_qa_agent import DEFAULT_TOOLS, GraphAgent

RUNS_PER_CASE = 10
INTER_CASE_DELAY = 0.1
RESULTS_DIR = Path("results")


# (tier, prompt, expected_tool, param_check).
# - expected_tool is None when no tool should be called.
# - "either:a|b" accepts either option.
CASES = [
    # Tier 1: obvious trigger words
    (1, "跑一次睡眠 pass", "trigger_sleep_pass", None),
    (1, "Run a sleep pass now", "trigger_sleep_pass", None),
    (1, "现在图里有多少节点？", "get_graph_stats", None),
    (1, "How big is the graph?", "get_graph_stats", None),

    # Tier 2: discriminate between similar tools
    (2, "最近合并了哪些实体？", "list_recent_merges", None),
    (2, "给我看看最近 5 次合并", "list_recent_merges", lambda a: a.get("k") == 5),
    (2, "我想知道 Tesla 这家公司的 CEO 是谁", "graph_query", None),
    (2, "图里关于新能源政策有什么信息？", "graph_query", None),

    # Tier 3: parameter extraction
    (3, "显示节点 abc-123-def 的来源", "show_provenance", lambda a: "abc-123-def" in str(a.get("node_or_edge_id", ""))),
    (3, "帮我查询：苹果和富士康之间有什么关系？", "graph_query",
        lambda a: "苹果" in a.get("question", "") and "富士康" in a.get("question", "")),
    (3, "列出最近 20 条合并", "list_recent_merges", lambda a: a.get("k") == 20),

    # Tier 4: shouldn't call any tool
    (4, "你好", None, None),
    (4, "谢谢", None, None),
    (4, "你能做什么？", None, None),
    (4, "解释一下什么是知识图谱", None, None),

    # Tier 5: synonymous-tool discrimination
    (5, "最近删了哪些边？", "list_recent_prunings", None),
    (5, "让我看看 pass 里裁掉了什么", "list_recent_prunings", None),
    (5, "最近清理了哪些节点？", "list_recent_prunings", None),
    (5, "最近合并了什么？", "list_recent_merges", None),

    # Tier 6: colloquial / short
    (6, "图多大？", "get_graph_stats", None),
    (6, "节点数", "get_graph_stats", None),
    (6, "睡一下", "trigger_sleep_pass", None),
    (6, "do a consolidation", "trigger_sleep_pass", None),

    # Tier 7: noisy parameter extraction
    (7, "嗯... 那个, 你帮我看看 node-xyz-789 是从哪来的好吧", "show_provenance",
        lambda a: "node-xyz-789" in str(a.get("node_or_edge_id", ""))),
    (7, "给我最近 3 条合并就够了", "list_recent_merges", lambda a: a.get("k") == 3),
    (7, "我想知道 edge id 是 e_42 的来源", "show_provenance",
        lambda a: "e_42" in str(a.get("node_or_edge_id", ""))),

    # Tier 8: double intent, accept either
    (8, "tesla 的 ceo 是谁？顺便告诉我图多大", "either:graph_query|get_graph_stats", None),

    # Tier 9: trap — looks like a write request but no write tool exists.
    # Refusal is the only correct answer (preparatory queries don't count
    # because there's nowhere for them to lead).
    (9, "帮我把节点 abc 的 summary 改成 'new text'", None, None),
    (9, "创建一个新节点叫 'TestNode'", None, None),
    (9, "导出整个图为 JSON", None, None),
    (9, "你能直接改 ontology 吗？", None, None),

    # Tier 10: chit-chat + half intent
    (10, "我今天想跑个 sleep pass, 但还没决定", None, None),
    (10, "你能跑 sleep pass 吗？", "trigger_sleep_pass", None),
    (10, "graph_query 这个工具是干嘛的？", None, None),
]


def _normalize_alternatives(expected, check):
    """Return list of (tool, check) alternatives regardless of input shape."""
    if isinstance(expected, str) and expected.startswith("either:"):
        tokens = expected.replace("either:", "").split("|")
        tools = [None if t == "None" else t for t in tokens]
        return [(t, check) for t in tools]
    return [(expected, check)]


def _tool_matches(actual: str | None, expected) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, str) and expected.startswith("either:"):
        tokens = expected.replace("either:", "").split("|")
        return actual in [None if t == "None" else t for t in tokens]
    return actual == expected


def _score_case(agent: GraphAgent, tier: int, prompt: str, expected, check) -> dict:
    alternatives = _normalize_alternatives(expected, check)
    r = {"tool_ok": 0, "param_ok": 0, "total": RUNS_PER_CASE,
         "errors": 0, "wrong": defaultdict(int), "sample_text": None}
    for _ in range(RUNS_PER_CASE):
        res = agent.route(prompt)
        if res.error:
            r["errors"] += 1
            continue
        tool = res.tool_name
        args = res.tool_args or {}
        matched = False
        for exp_tool, exp_check in alternatives:
            if _tool_matches(tool, exp_tool):
                r["tool_ok"] += 1
                if exp_tool is None or exp_check is None or (args and exp_check(args)):
                    r["param_ok"] += 1
                matched = True
                break
        if not matched:
            r["wrong"][str(tool)] += 1
            only_none = all(e[0] is None for e in alternatives)
            if only_none and res.text and not r["sample_text"]:
                r["sample_text"] = res.text[:100]
        time.sleep(INTER_CASE_DELAY)
    return r


def run() -> str:
    gi = GraphInstance("test-toolcalling", "data/experiment", "ontology.md")
    agent = GraphAgent(gi, expose_tools=DEFAULT_TOOLS)

    lines: list[str] = []
    header = (
        f"=== Tool-calling probe ===\n"
        f"exposed tools ({len(DEFAULT_TOOLS)}): {', '.join(DEFAULT_TOOLS)}\n"
        f"runs/case: {RUNS_PER_CASE} | temp=0.2 | "
        f"model: {agent.client.model}\n"
    )
    lines.append(header)
    print(header)

    tier_sum: dict[int, dict] = defaultdict(lambda: {"tool_ok": 0, "param_ok": 0, "total": 0})

    for tier, prompt, expected, check in CASES:
        r = _score_case(agent, tier, prompt, expected, check)
        tier_sum[tier]["tool_ok"] += r["tool_ok"]
        tier_sum[tier]["param_ok"] += r["param_ok"]
        tier_sum[tier]["total"] += r["total"]

        status = "OK" if r["tool_ok"] == r["total"] else ("PARTIAL" if r["tool_ok"] > 0 else "FAIL")
        line = (f"[T{tier}] {status:<7} tool {r['tool_ok']}/{r['total']}  "
                f"param {r['param_ok']}/{r['total']}  | {prompt[:55]}")
        lines.append(line)
        print(line)
        if r["wrong"]:
            wline = f"         wrong calls: {dict(r['wrong'])}"
            lines.append(wline)
            print(wline)
        if r["sample_text"]:
            sline = f"         reject-reply sample: {r['sample_text']}"
            lines.append(sline)
            print(sline)

    labels = {1: "基础", 2: "多工具", 3: "参数", 4: "边界", 5: "近义", 6: "口语", 7: "噪声", 8: "多意图", 9: "陷阱", 10: "元意图"}
    summary = ["", "====== Tier summary ======",
               f"{'Tier':<14} {'Tool acc':<12} {'Param acc':<12}"]
    for tier in sorted(tier_sum.keys()):
        s = tier_sum[tier]
        tool_acc = s["tool_ok"] / s["total"] if s["total"] else 0
        param_acc = s["param_ok"] / s["total"] if s["total"] else 0
        summary.append(f"T{tier} {labels.get(tier, ''):<10} {tool_acc:>6.1%}      {param_acc:>6.1%}")
    summary.append("")
    summary.append(
        "Tier-9 rubric note: knowledge enters the graph only through "
        "`ingest_file` against documents under data/raw/. Direct "
        "create-node / edit-summary requests have no matching tool, "
        "so refusal (no tool call) is the only correct answer."
    )
    summary_text = "\n".join(summary)
    print(summary_text)
    lines.append(summary_text)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "tool_calling.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWritten: {out}")
    return "\n".join(lines)


def main():
    run()


if __name__ == "__main__":
    main()
