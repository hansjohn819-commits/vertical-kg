# Vertical-Domain Self-Evolving Knowledge Graph

Northeastern ALY 6980 Capstone project. A knowledge graph system that evolves via periodic "sleep passes" — pruning, entity consolidation, weight reinforcement, and link formation — in order to surface *derived* knowledge that cannot be read off any single raw source, user contribution, or point in time.

## Status

Early-stage scaffolding. Data not yet in; demo is the minimum end-to-end skeleton: ingest → graph → Q&A agent → online write → (stub) sleep pass → Streamlit chat.

## Architecture (brief)

Four modules, three-layer data separation:

- **M1** Graph Construction — raw → initial graph
- **M2** Q&A Agent — LangGraph agent with a toolbox (query + management tools)
- **M3** Live Integration — online writes from conversation turns
- **M4** Sleep Pass — periodic maintenance: reinforce → consolidate → prune → link-form

```
workspace/
├── data/              # raw / production / experiment instances (gitignored)
├── ontology.md        # schema layer (entity & relation types, merge rules)
├── src/
│   ├── graph/         # Node / Edge / Provenance / GraphInstance
│   ├── modules/       # m1, m2, m3, m4_sleep_pass/
│   ├── evaluation/    # qualitative + plant-and-recover + baseline RAG
│   ├── llm/           # routing + local OpenAI-compatible client
│   └── dashboard/     # streamlit_app.py
└── tests/
```

## Running the demo

```bash
pip install -r requirements.txt
cp .env.example .env     # edit LOCAL_LLM_BASE_URL, model name
streamlit run src/dashboard/streamlit_app.py
```

Requires a local OpenAI-compatible LLM endpoint (e.g. a local server hosting `google/gemma-4-26b-a4b`).

## Testing philosophy

User interaction is **manually driven via the Streamlit chat** — real-world samples are pasted in by the user to simulate distributed, cross-time user behavior. There is no internal simulated-user harness.

## Phase 6 — Sleep pass (M4)

The sleep pass is modeled as a LangGraph `StateGraph` over four sub-operations in the fixed order `4c → 4b → 4a → 4d` (guide §5). 4c is one-shot; 4b / 4a / 4d each live in a self-looping node that converges before the graph hands control to the next.

Convergence rules (guide §5.0 + user direction):

- **4b merge** — after each round, a fresh-memory LLM explicitly votes `stop` or `continue`. Hard cap `MERGE_MAX_ITER=10`.
- **4a prune** — mechanical: round with no new deletion → exit.
- **4d link-form** — mechanical: round with no new passing edge → exit. Pairs judged once are remembered per-pass so we do not re-ask rejections.

Candidate generation for 4b combines (a) embedding top-k (`sentence-transformers/all-MiniLM-L6-v2`, reused from M2 retrieval), (b) neighbor-set Jaccard ≥ 0.3, and (c) same-type-with-cos ≥ 0.55 as documented in §5.5 "同类型也进候选". 4d seeds are this pass's merges plus the top-decile reinforced nodes, BFS to depth 2. 4d strictly drops any pair where the LLM cannot state both *what* the relation is and *why* it holds (§5.7 critical rule).

`trigger_sleep_pass` runs synchronously. While a pass is in flight the M2 agent rejects new requests with `"Sleep pass is currently running"` — by design, the consolidation is the centerpiece, not chat availability.

Smoke test (toy 12-node graph with Tesla Inc / Tesla Motors planted as a merge target):

```bash
python -m tests.test_sleep_pass
```

One full pass against the toy seed currently:

- merges Tesla Inc + Tesla Motors into a single node (LLM reasoning cites the 2017 legal rename);
- prunes a low-weight stale edge pre-marked suspicious;
- adds one derived `MANUFACTURER_IN` edge Tesla → Electric Vehicles via 4d 2-hop BFS.

## Phase 5 — Tool-calling validation

Two-round routing test over 35 prompts × 10 runs × temp=0.7 on `google/gemma-4-26b-a4b`, driven through the real `GraphAgent` (not a parallel tool array). Tiers follow the 2026-04-20 baseline at `F:\Master_CA\Python Test\test.py`.

- **Round 1** — 9 tools exposed (read-only + evaluation). Write tools (`upsert_node`, `upsert_edge`) withheld on purpose; tier-9 "create a node" trap expects refusal. Strict rubric.
- **Round 2** — all 11 tools exposed. Write-adjacent tier-9 cases use a lenient rubric that also accepts `graph_query` / `read_ontology` as preparatory check-before-write steps.

| Tier | Round 1 (tool / param) | Round 2 (tool / param) |
|---|---|---|
| T1 基础 | 100% / 100% | 100% / 100% |
| T2 多工具 | 100% / 100% | 100% / 100% |
| T3 参数 | 100% / 100% | 100% / 100% |
| T4 边界 | 100% / 100% | 100% / 100% |
| T5 近义 | 100% / 100% | 100% / 100% |
| T6 口语 | 100% / 100% | 100% / 100% |
| T7 噪声 | 100% / 100% | 100% / 100% |
| T8 多意图 | 100% / 100% | 100% / 100% |
| T9 陷阱 | **95.0% / 95.0%** | **97.5% / 97.5%** |
| T10 元意图 | 100% / 100% | 100% / 100% |

Findings:

1. **Exposing write tools did not pollute routing on other tiers** — T1–T8 and T10 stay at 100% in both rounds.
2. **"Check-before-write" is a stable tendency**. In round 2, the "edit summary" prompt goes through `graph_query` 10/10 before attempting a write. Round 1 also shows 1/10 agents taking the same verify path even with no write tool available. This mirrors the verify-then-write pattern that M4 sleep pass uses under the LangGraph pass-within convergence loop (§5.0 option β).
3. **Write tools are not exposed to chat in production.** `upsert_node` / `upsert_edge` are driven by M4 sleep pass. The round-2 lenient rubric reflects this: a preparatory query is not a failure, it is the same pattern the sleep pass itself follows.

Rerun:

```bash
python -m tests.test_tool_calling both   # or 1 / 2
# Results at results/round{1,2}.txt (gitignored)
```

## License

TBD.
