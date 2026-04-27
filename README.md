# Vertical-Domain Self-Evolving Knowledge Graph

A vertical-domain knowledge graph that evolves autonomously via periodic "sleep passes" — pruning, entity consolidation, weight reinforcement, and link formation — to surface *derived* knowledge that cannot be read off any single raw source, user contribution, or point in time. The goal is a graph whose value grows after ingestion, not just during it.

## Status

End-to-end skeleton in place: ingest → graph → Q&A agent → online write → sleep pass → Streamlit chat. Real-domain data validation is the next milestone.

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

## Dashboard

Single-file Streamlit app at `src/dashboard/streamlit_app.py` wraps `GraphAgent` as a chat surface. All capabilities — Q&A, stats, ingestion, sleep-pass trigger, provenance lookup, merge/prune history, ontology read, entity write — are exposed implicitly through the agent's tool box; the user asks in natural language, the agent picks the tool.

Chat history is fed back to the agent through a rolling token window (`HISTORY_BUDGET_TOKENS=3000`), so prompt size stays bounded regardless of session length. Storage is persisted after every turn. While a sleep pass is in flight, the top banner flips to a pause notice and the agent short-circuits new requests — consolidation is the centerpiece, not chat availability.

## Core design: the sleep pass

The sleep pass is modeled as a LangGraph `StateGraph` over four sub-operations in the fixed order `4c → 4b → 4a → 4d`. 4c (weight reinforcement) is one-shot; 4b (merge), 4a (prune), and 4d (link formation) each live in a self-looping node that converges before the graph hands control to the next.

Convergence rules:

- **4b merge** — after each round, a fresh-memory LLM explicitly votes `stop` or `continue`. Hard cap `MERGE_MAX_ITER=10`.
- **4a prune** — mechanical: round with no new deletion → exit.
- **4d link-form** — mechanical: round with no new passing edge → exit. Pairs judged once are remembered per-pass so we do not re-ask rejections.

Candidate generation for 4b combines (a) embedding top-k (`sentence-transformers/all-MiniLM-L6-v2`, reused from M2 retrieval), (b) neighbor-set Jaccard ≥ 0.3, and (c) same-type-with-cos ≥ 0.55 (same-type pairs also qualify as candidates). 4d seeds are this pass's merges plus the top-decile reinforced nodes, BFS to depth 2. 4d strictly drops any pair where the LLM cannot state both *what* the relation is and *why* it holds.

`trigger_sleep_pass` runs synchronously; the M2 agent rejects new requests while it runs.

Smoke test (toy 12-node graph with Tesla Inc / Tesla Motors planted as a merge target):

```bash
python -m tests.test_sleep_pass
```

One full pass against the toy seed currently:

- merges Tesla Inc + Tesla Motors into a single node (LLM reasoning cites the 2017 legal rename);
- prunes a low-weight stale edge pre-marked suspicious;
- adds one derived `MANUFACTURER_IN` edge Tesla → Electric Vehicles via 4d 2-hop BFS.

## Tool-calling validation

Two-round routing test over 10 prompts × 10 runs × temp=0.7 on `google/gemma-4-26b-a4b`, driven through the real `GraphAgent` (not a parallel tool array). Tiers follow an external baseline test harness.

- **Round 1** — 9 tools exposed (read-only + evaluation). Write tools (`upsert_node`, `upsert_edge`) withheld on purpose; tier-9 "create a node" trap expects refusal. Strict rubric.
- **Round 2** — all 11 tools exposed. Write-adjacent tier-9 cases use a lenient rubric that also accepts `graph_query` / `read_ontology` as preparatory check-before-write steps.

| Tier | Round 1 (tool / param) | Round 2 (tool / param) |
|---|---|---|
| T1 Basic | 100% / 100% | 100% / 100% |
| T2 Multi-tool | 100% / 100% | 100% / 100% |
| T3 Parameter | 100% / 100% | 100% / 100% |
| T4 Edge case | 100% / 100% | 100% / 100% |
| T5 Synonym | 100% / 100% | 100% / 100% |
| T6 Colloquial | 100% / 100% | 100% / 100% |
| T7 Noise | 100% / 100% | 100% / 100% |
| T8 Multi-intent | 100% / 100% | 100% / 100% |
| T9 Trap | **95.0% / 95.0%** | **97.5% / 97.5%** |
| T10 Meta-intent | 100% / 100% | 100% / 100% |

Findings:

1. **Exposing write tools did not pollute routing on other tiers** — T1–T8 and T10 stay at 100% in both rounds.
2. **"Check-before-write" is a stable tendency**. In round 2, the "edit summary" prompt goes through `graph_query` 10/10 before attempting a write. Round 1 also shows 1/10 agents taking the same verify path even with no write tool available. This mirrors the verify-then-write pattern that M4 sleep pass uses under its LangGraph convergence loop.
3. **Write tools are not exposed to chat in production.** `upsert_node` / `upsert_edge` are driven by M4 sleep pass. The round-2 lenient rubric reflects this: a preparatory query is not a failure, it is the same pattern the sleep pass itself follows.

Rerun:

```bash
python -m tests.test_tool_calling both   # or 1 / 2
# Results at results/round{1,2}.txt (gitignored)
```

## License

MIT — see [LICENSE](LICENSE).
