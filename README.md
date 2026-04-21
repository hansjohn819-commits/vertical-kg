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

## License

TBD.
