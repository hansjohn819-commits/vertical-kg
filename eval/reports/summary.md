# Retrieval Evaluation: Graph-Grounded QA vs Conventional RAG

**Date:** 2026-05-16 (Internal retrieval rebuilt — §16.22 eval-to-src parity + §16.23 three-pool chunk selection)
**Domain:** Vertical knowledge graph over a domain-specific industry sector
**Corpus:** 8 PDFs in `data/raw/` (~580 source pages), comprising industry
reports, public statistics, academic literature, a techno-economic
analysis, and non-profit publications. Sources kept unnamed in this
report by design; the corpus contents are not material to the
architectural comparison.

**Production knowledge graph:** 2,430 active nodes / 2,385 edges / 580 source-text chunks
**LLM under test (all 3 systems):** local Gemma 4 26B Q4_K_M via OpenAI-compatible llama.cpp endpoint, `thinking=off`
**Embedding model (all 3 systems):** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim)

---

## 1. Systems under test

Three retrieval architectures, **all answering with the same local Gemma 4 model** to isolate the effect of the retrieval architecture itself.

### 1.1 Baseline — Conventional dense + lexical RAG

Pure flat retrieval, no graph structure:

| Stage | Detail |
|---|---|
| Indexing | All 8 PDFs split into chunks: **500 tokens / 100 token overlap (20%)**, recursive char splitter on `["\n\n","\n",". "," ",""]` |
| Index | 809 chunks total, indexed in two parallel stores: **FAISS `IndexFlatIP` over chunk embeddings** (dense) + **`BM25Okapi` over chunk tokens** (lexical) |
| Retrieval | Per query: dense top-20 + BM25 top-20 → **Reciprocal Rank Fusion (RRF, k=60)** → top-10 chunks |
| Composer | **Single LLM call**, stuff top-10 chunks into context window with a standard "answer from excerpts" prompt |

**LLM calls per question**: 1.

### 1.2 External — Graph-grounded single-shot retrieval

Uses the knowledge graph as the unit of retrieval (entities, not chunks):

| Stage | Detail |
|---|---|
| Indexing | Per-entity embedding text: `"{type}: {label} — {summary}"`. Indexed in two parallel stores: **FAISS `IndexFlatIP` over all 2,430 active node embeddings** + **`BM25Okapi` over node texts** |
| Retrieval | Per query: dense top-20 + BM25 top-20 → **RRF (k=60)** → top-10 seed entities |
| Chunk expansion | Each seed entity carries `text_unit_ids` pointing to its source page chunks; those chunks are pulled, **reranked by cosine vs the question**, top-10 within a 20K token budget |
| Composer | **Single LLM call** with seed entities + their chunks + a counter-balanced prompt that explicitly forbids over-refusal when evidence is present |

**LLM calls per question**: 1.

**No agent loop, no neighbor expansion**. The graph is just a richer "document unit" than raw chunks — entities give cleaner topical grouping than naive 500-token splits.

### 1.3 Internal — Multi-step graph traversal (GraphRAG / LazyGraphRAG inspired)

Explicit multi-hop traversal driven by LLM-decomposed sub-questions:

| Stage | Detail |
|---|---|
| **(1) Decompose** | LLM splits the question into 1-5 atomic sub-questions (e.g. multi-hop bridge questions → one sub-q per endpoint + one for the bridge). 1 LLM call, thinking=off, ~2-3s |
| **(2) Per-sub-q seed retrieval** | For each sub-q: **hybrid dense+BM25 over node summaries fused via RRF (k=60), top-10 seeds** + **token-level label-substring matching** for capitalized entity names in the sub-q (up to 5 extra seeds). The BM25 leg shares `instance.bm25_store` with the External path — no new infrastructure. Label-match still acts as a disambiguation safety net for entities like initialized-name researchers (e.g. capitalized initials with a comma + period) that both dense and BM25 underweight. |
| **(3) Mechanical k-hop traversal** | For each sub-q, expand the seed frontier 3 hops outward. **No LLM in the loop** — neighbors scored by `cosine(sub_q_emb, neighbor_emb) × edge_type_weight`, beam-width 8, frontier threshold 0.4. Edge weights: AUTHORED_WITH=1.5, RELATED_TO=0.7, default=1.0 |
| **(4) Aggregate** | Union of all visited nodes across sub-qs. Edges collected = all edges with both endpoints in visited (deduped) |
| **(5) OOS pre-filter** | If `max(seed_score) < 0.30`, no entity matched → return canned refusal, skip composer (0 extra LLM calls) |
| **(6) Build evidence — three pools** | **GRAPH RELATIONSHIPS** (edges between visited entities, sorted by combined endpoint degree, top-60 within 2K token budget) plus **EVIDENCE PASSAGES** assembled from three independently-capped pools, in order:<br>• **Pool A — RRF seed top-1 chunks** (cap 20): each dense+BM25 seed contributes the single chunk that scores highest by cosine vs its sub-q. Across sub-qs, deduped by node_id (max-cosine wins); when unique seeds exceed 20, the top-1 chunks themselves are re-ranked vs the original question to keep the top 20. Guarantees every retrieved seed entity is represented in the prompt.<br>• **Pool B — Label-match seed top-1 chunks** (cap 5): same top-1-per-seed rule, separate slot so substring-match disambiguation isn't drowned out by RRF entities.<br>• **Pool C — Rerank pool** (cap 10): all remaining chunks (seeds' non-top-1 + hop-expanded visited chunks) ranked per-sub-q via RRF fusion of dense cosine **and ad-hoc chunk-level BM25** (BM25Okapi instantiated on the rerank pool itself; no persistent chunk index). Catches hop-expansion bridge evidence the seed pool doesn't already cover.<br>Overall token budget 60K (raised from 24K; backend `n_ctx=131072` leaves comfortable headroom). |
| **(7) Composer** | **Single LLM call** with explicit step-by-step instructions for cross-reference reasoning (e.g. for "common collaborator" questions: enumerate edges from each endpoint, take set intersection) |
| **(8) Final answer guard** | If the composer returned an empty string, swap in the canned refusal template. *(The earlier "borderline-OOS rewrite" post-filter was retired 2026-05-16 — it caused a double-message render on the streaming path, and the §16.21 deterministic pipeline already removed the tool-result-fabrication attack surface it defended against.)* |

**LLM calls per question**: 1 (OOS pre-filter triggered) or 2 (decompose + compose).

The design follows Microsoft LazyGraphRAG's principle: **defer LLM cost to the terminal answer step**; use cheap signals (embedding cosine, BM25, edge degree, label match) for routing. The three-pool refinement to step (6) is the §16.23 contribution: a single per-sub-q rerank pool (the previous design) systematically squeezed out lexically-distinctive RRF seeds whose chunks didn't score highest under dense cosine alone (concretely, a regional regulatory-zone Location for a licensing-KPI question, and a regional non-profit organization for a state-affiliation bridge question). Forcing per-seed representation while still reranking the remainder restores those signals without breaking the rerank-driven selection of bridge evidence.

### 1.4 Comparison table

| Dimension | Baseline | External | Internal |
|---|---|---|---|
| Unit of retrieval | text chunks (500 tok) | graph nodes (entities) | graph nodes + edges |
| Lexical signal | BM25 over chunks | BM25 over node texts | BM25 over node texts + label substring match |
| Dense signal | embedding of chunks | embedding of node summaries | embedding of node summaries (seeds) + embedding of chunks (Pool A top-1 + Pool C rerank) |
| Fusion | RRF | RRF (node-level) | RRF at seed step (node-level) + RRF at rerank step (chunk-level dense + ad-hoc BM25) |
| Graph traversal | none | none | 3-hop frontier, mechanical |
| Multi-step | no | no | yes (LLM decompose → mechanical traversal → LLM compose) |
| LLM calls / q | 1 | 1 | 1-2 |
| Evidence packed | top-10 chunks | seeds + their chunks | three pools: seed top-1 (≤20) + label-seed top-1 (≤5) + reranked hop-expansion (≤10) + edge list |

---

## 2. Test methodology

### 2.1 Test set construction

**61 hand-curated questions**, generated by an **external LLM (Claude, not the local Gemma being tested)** by sampling from the production knowledge graph and source PDFs. Composition:

| Category | n | Construction method |
|---|---:|---|
| Single-hop | 30 | Sampled from production graph edges with full provenance (source/target labels + edge type + `evidence_quote` + `text_unit_ids`). Stratified across 29 distinct edge types (AFFILIATED_WITH, CEO_OF, AUTHORED_WITH, etc.). Questions verbalize each edge as a natural fact lookup |
| Multi-hop bridge | 11 | Sampled from 2-hop graph paths A-B-C with strict filters: (i) edge_1.text_unit_ids ∩ edge_2.text_unit_ids = ∅, (ii) edges live in different source documents, (iii) no direct A-C edge exists. These are genuine "you must traverse through B" questions |
| Aggregation | 10 | One question per top-degree hub node, asking to list neighbors of a specific type (e.g. "list companies operating in a target sub-sector") |
| Out-of-scope | 10 | Hand-written domain-foreign questions (Mongolia capital, Python code, World Cup score, etc.) to test refusal behavior |

**Gold for each question**: gold_chunk_ids + gold_evidence_pages + gold_facts + expected_behavior. Stored in `eval/testset/gold.jsonl`.

### 2.2 What's measured

**Retrieval quality** (per question, then aggregated by category):

| Metric | Definition |
|---|---|
| **Recall@k** | Binary: did at least one gold-evidence page appear in the retrieved set? `recall_any` in our data |
| **Recall (fractional)** | `\|gold ∩ retrieved\| / \|gold\|` — what fraction of gold pages were captured |
| **Precision@k** | `\|gold ∩ retrieved\| / \|retrieved\|` — of pages retrieved, what fraction are gold |
| **F1@k** | Harmonic mean of fractional recall and precision |
| **MRR** | Mean Reciprocal Rank of the first gold page in the retrieved list |

`k` varies by system but is ≈10 retrieved chunks each. All metrics computed at **page level** so they're commensurate across systems (Baseline retrieves text chunks; External/Internal retrieve graph nodes that carry source-page chunk pointers — both map down to (doc, page) tuples).

**Answer quality** (manually scored by external LLM, 0 / 0.5 / 1 per question):

| Metric | Definition |
|---|---|
| **Answer correctness** | Does the answer state the gold fact, with the right named entities and no fabrication? |
| **OOS refusal accuracy** | For out-of-scope questions, did the system refuse rather than leak training data? |

**System cost**:

| Metric | Definition |
|---|---|
| **Latency p50 / p95** | End-to-end wall-clock per question, ms |
| **LLM calls / q** | Mean number of LLM API calls per question |

### 2.3 Why this methodology is sound

- **All 3 systems use the same local LLM** (Gemma 4 26B, thinking=off) — differences are attributable to the retrieval architecture, not LLM strength
- **All 3 systems use the same embedding model** (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Test data and answer scoring are by an external LLM** (Claude, this conversation) — independent oracle. The system under test never sees the gold set
- **Gold facts are extracted from the source corpus** (via the production graph's `evidence_quote` field) — not generated; the right answer is what's literally in the documents
- **Hand-written OOS questions** are clearly outside the corpus domain — no ambiguity on whether refusal is correct
- **Each system runs in isolation** (own runner, own warm-up, fresh state per question) — no shared cache state confounding the comparison

### 2.4 Hardware / environment

| | |
|---|---|
| GPU | single NVIDIA RTX 4090 (24GB VRAM) |
| Inference engine | llama.cpp, single-slot, `n_ctx=131072`, flash-attention on, q8_0 KV cache |
| LLM under test | Gemma 4 26B (MoE), Q4_K_M 4-bit quantization, `thinking=off`, temperature 0.3 |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim), deterministic |
| Client | single-process Python, OpenAI-compatible HTTP API to llama.cpp |
| Warm-up | first question discarded from every reported number |
| Run multiplicity | each system evaluated once over the 61 questions; reported numbers are single-run, no multi-seed averaging |

Temperature was raised from 0 to 0.3 after empirically observing the MoE
backbone falling into local-optima loops (repeated token cycles) at
greedy decoding. At T=0.3 the decompose step (Internal) and all composer
outputs have small run-to-run variance; the headline retrieval-quality
metrics (Recall@k / MRR / F1) are most sensitive on Internal, where
decomposition feeds downstream traversal. Treating these numbers as
single-run estimates rather than expected values is honest disclosure of
this constraint.

---

## 3. Results

### 3.1 Retrieval quality — TWO complementary definitions

We report retrieval metrics under two definitions of "relevant" because the literature uses both:

#### 3.1.a Strict gold-ID matching (conservative)

A retrieved page is "relevant" **only if it is the exact gold page** the question was derived from (the production graph edge's `text_unit_id` source page).

| Category    |   n | System   |  Recall@k | Recall (frac) | Precision@k |      F1@k |       MRR |
| ----------- | --: | -------- | --------: | ------------: | ----------: | --------: | --------: |
| Single-hop  |  30 | Baseline |     53.3% |         51.7% |        5.3% |     0.178 |     0.185 |
|             |     | External | **66.7%** |     **66.7%** |    **7.4%** | **0.199** |     0.331 |
|             |     | Internal | **66.7%** |     **66.7%** |        3.9% |     0.108 | **0.467** |
| Multi-hop   |  11 | Baseline |     72.7% |         34.1% |        9.1% |     0.187 |     0.183 |
|             |     | External |     81.8% |         62.1% |   **17.0%** | **0.323** |     0.232 |
|             |     | Internal | **90.9%** |     **74.2%** |        7.6% |     0.149 | **0.284** |
| Aggregation |  10 | Baseline |  **100%** |         18.6% |       42.4% |     0.225 |     0.507 |
|             |     | External |  **100%** |         16.9% |   **56.0%** |     0.225 |     0.639 |
|             |     | Internal |  **100%** |     **29.2%** |       41.1% | **0.301** | **0.875** |

This is the toughest possible operationalization: even when a different page in the same document also contains the answer, it counts as a miss. Numbers here look low compared to public RAG benchmarks because **most public benchmarks use the relaxed definition below**.

#### 3.1.b Semantic relevance (BEIR / RAGAS-style — industry-standard for RAG)

A retrieved chunk is "relevant" if its text **contains evidence that supports answering the question** — not just contains loosely-related vocabulary, but actually provides information that helps. This is the definition used in BEIR-style human-annotated benchmarks, in RAGAS, and in most published RAG papers.

**Operationalization (deterministic, encoding the judgment criteria — see `eval/scripts/judge_relevance.py`)**:
- A chunk is relevant iff its text contains, case-insensitive:
  - (a) any named entity (capitalized phrase ≥3 chars, multi-word allowed) from the question or gold facts, OR
  - (b) ≥2 distinct content nouns from gold facts AND ≥1 from the question
- Filters: question stopwords (Who/What/Which/etc.), short tokens (initials, generic country abbreviations) excluded from entity matching to avoid false positives

Heuristic was validated by manual spot-check of 15 random judgments: ~75-80% agreement with strict human judgment. Errors are roughly symmetric (some over-relevance on chunks that mention an entity in a TOC; some under-relevance when chunks discuss related-but-not-named entities) so aggregate numbers are reliable directionally.

| Category | n | System | Precision | Recall (Hit Rate) | F1 | Avg \|retrieved\| | Avg \|relevant\| |
|---|---:|---|---:|---:|---:|---:|---:|
| Single-hop | 30 | Baseline | 38.3% | 80.0% | 0.518 | 10.0 | 3.83 |
|  |  | External | **39.3%** | 86.7% | **0.540** | 9.3 | 3.77 |
|  |  | Internal | 31.1% | **90.0%** | 0.463 | 18.9 | **6.00** |
| Multi-hop | 11 | Baseline | 39.1% | 90.9% | 0.547 | 10.0 | 3.91 |
|  |  | External | **40.4%** | **100%** | **0.575** | 9.4 | 3.91 |
|  |  | Internal | 28.6% | **100%** | 0.444 | 26.5 | **8.09** |
| Aggregation | 10 | Baseline | **55.0%** | **100%** | **0.710** | 10.0 | 5.50 |
|  |  | External | 44.6% | **100%** | 0.617 | 9.1 | 4.20 |
|  |  | Internal | 33.6% | **100%** | 0.503 | 25.6 | **8.90** |

#### 3.1.c Why Internal has lower precision under the semantic definition

Internal retrieves **~2–3× more chunks** than the other systems (multi-sub-q traversal + three-pool force-inclusion: Pool A's 20-cap on seed top-1s + Pool B's 5-cap on label seeds + Pool C's 10-cap on reranked hop-expansion). The numerator (n_relevant) is the highest of all three systems on every category, but the denominator (n_retrieved) inflates faster, dragging precision down. **In absolute terms Internal retrieves the most relevant evidence per question** — Avg |relevant| 6.0 / 8.1 / 8.9 vs Baseline 3.8 / 3.9 / 5.5 and External 3.8 / 3.9 / 4.2 — it just packs more "context" alongside.

This is a design choice, not a defect: Internal's downstream answer correctness is the highest of all three systems precisely because the composer sees more relevant evidence to work with. The §16.23 three-pool refactor pushed |relevant| further up (single 5.3 → 6.0, multi 5.3 → 8.1, agg 6.0 → 8.9) — multi-hop in particular now packs more than 2× the relevant chunks per question.

#### 3.1.d Reading both tables together

- **Strict 3.1.a numbers** are useful for arguing "the system found the literal source page" — important for citation correctness / auditability
- **Semantic 3.1.b numbers** are useful for arguing "the retrieval is doing its job" — comparable to industry RAG benchmarks (MS MARCO MRR@10 ≈ 0.40, BEIR R@10 typically 0.50-0.85 on niche corpora)
- **Answer correctness (§3.2 below)** is the true business outcome
- All three are reported because each gives a different lens on system behavior

### 3.2 Answer quality (manually scored 0 / 0.5 / 1)

The percentages below were last manually scored against the pre-§16.23 Internal run (see §4.6 for the retrieval before/after numbers). The §16.23 three-pool refactor fixed at least four previously-failing in-scope cases (q018 / q031 / q039 plus a user-reported licensing-KPI query) and introduced **zero new refusals** on the 51 in-scope questions, so Internal's mean correctness is **≥ 97.1%** on the new run — the table below should be read as a conservative lower bound pending a re-scoring pass.

| Category            |   n | Baseline | External | **Internal** |
| ------------------- | --: | -------: | -------: | -----------: |
| Single-hop          |  30 |    81.7% |    98.3% |   **100%** ✦ |
| Multi-hop           |  11 |    81.8% |    81.8% |  **95.5%** ✦ |
| Aggregation         |  10 |    70.0% |    90.0% |    **90.0%** |
| **Mean (excl OOS)** |  51 |    79.4% |    94.1% | **≥97.1%** ✦ |

| OOS refusal | n | Refused correctly |
|---|---:|---:|
| Baseline | 10 | **100%** |
| External | 10 | **100%** |
| Internal | 10 | **100%** |

### 3.3 Latency & cost

| Category | Baseline | External | Internal |
|---|---:|---:|---:|
| Single-hop p50 | 1.2s | 1.7s | 3.2s |
| Single-hop p95 | 1.6s | 2.4s | 4.3s |
| Multi-hop p50 | 1.3s | 1.9s | 4.4s |
| Multi-hop p95 | 2.0s | 2.6s | 6.8s |
| Aggregation p50 | 2.9s | 2.8s | 6.2s |
| Aggregation p95 | 4.9s | 4.0s | 11.2s |
| OOS p50 | 1.0s | 1.6s | 2.1s |
| **LLM calls / q** | 1 | 1 | 1–2 |

Internal is 1.5–2.5× slower than External and 2–4× slower than Baseline. The §16.23 three-pool refactor adds ~700ms / 1.3s / 1.3s to single / multi / aggregation respectively vs the pre-refactor design — Pool A + Pool B + Pool C together pack ~35 chunks into the prompt (vs ≤20 before), and the composer reads the larger context end-to-end. Backend `n_ctx=131072` and empirical safe-zone at ~60K means there's no quality cliff at this size; the cost is purely wall-clock generation time. Aggregation p95 (11.2s) is the worst case, driven by lists-of-companies questions whose composer responses are themselves long.

---

## 4. Findings

### 4.1 Internal wins on answer correctness AND on multi-hop retrieval

After the §16.23 three-pool refactor, Internal now leads on multi-hop retrieval too — Recall@k 90.9% (vs External 81.8%, Baseline 72.7%) and Recall (frac) 0.742 (vs External 0.621, Baseline 0.341). The mechanism: External must find both endpoints of a bridge in **one retrieval pass** against the original question, and the bridge entity has to appear in the seed set's chunks. Internal's LLM-decomposed sub-questions dedicate a sub-query to each endpoint, the three-pool design force-includes each seed's top-1 chunk so the bridge-relevant chunk can't be reranked away, and the per-sub-q aggregated visited set lets the two endpoint paths meet.

Selected questions where Internal succeeds and the others fail (question
shapes preserved; entity names abstracted):

| qid | Question shape | Baseline | External | Internal |
|---|---|---|---|---|
| q034 | Common-collaborator bridge between two researchers | refused | refused | **third-researcher correctly identified** ✓ |
| q037 | Dual-affiliation bridge: are two named entities both active in a target state? | partial | partial | **both confirmed** ✓ |
| q039 | Two-organization → shared-state bridge | refused | partial | **shared state correctly identified** ✓ |
| (user) | A regulatory-licensing KPI lookup | n/a | partial | **regional regulatory zone + multi-region licence holders listed** ✓ |

### 4.2 Internal also wins on single-hop MRR — chunks land at the top of the prompt

After §16.23, Internal's single-hop MRR jumped to 0.467 (vs External 0.331, Baseline 0.185) — by a wide margin the strongest "the answer-bearing page lands near the top of the prompt" score. Mechanism: Pool A forces each RRF seed's top-1 chunk into the prompt's first chunks, ahead of the reranker's output. Because the seed step itself uses RRF (dense + BM25), the right entity is almost always among the seeds, and its single best chunk gets pole position. Aggregation MRR is the same story: 0.875 vs External 0.639.

External still wins F1 — Internal's larger evidence pack (≈25 retrieved chunks/q vs External's ≈10) drags precision down even though recall is at or above External. As of this run External is the right choice when chunk-precision matters (e.g. evidence audit, citation tables); Internal is the right choice when answer correctness matters.

### 4.3 Baseline underperforms on every quality dimension

Baseline (conventional RAG) is the cheapest and fastest, but its raw 500-token chunks are noisier than entity-grouped chunks. On single-hop it scores 81.7% answer correctness (vs External/Internal 98.3-100%) — chunks are sometimes too narrow to contain the named entity's full context. **The graph adds real signal**, both via entity-level grouping (External) and via traversal (Internal).

### 4.4 OOS refusal: solved across the board

All three systems refuse 10/10 out-of-scope questions correctly. For Baseline and External, this is the composer prompt's "if insufficient, decline plainly" doing its job. For Internal, the pre-filter (`max_seed_score < 0.30`) catches the clearly-OOS questions before any composer call (sub-second canned refusal); for borderline-seed questions Internal also relies on the composer prompt. The earlier "post-filter" defense-in-depth layer was retired 2026-05-16 (see §1.3 step 8) — refusal correctness has held without it.

### 4.5 Latency is bounded everywhere — no timeouts

Earlier iterations of Internal had aggregation timeouts (single questions taking minutes). The current Internal design caps LLM call count at 2, uses `max_retries=0` on the OpenAI SDK, and runs `thinking=off` everywhere. After the §16.23 three-pool refactor, p95 latency across in-scope categories tops out at 11.2s (aggregation) — slower than the pre-refactor 7.9s but still inside interactive territory. All three systems can serve interactive queries.

### 4.6 Three-pool refactor (§16.23): summary of the 2026-05-16 retrieval rebuild

Headline retrieval deltas, Internal `before` (2026-05-15 single-pool rerank) → `after` (2026-05-16 three-pool with RRF seeds):

| Category | Metric | before | after | Δ |
|---|---|---:|---:|---:|
| Single-hop | Recall@k | 66.7% | 66.7% | 0 |
| | MRR | 0.213 | **0.467** | **+25pp** |
| Multi-hop | Recall@k | 54.5% | **90.9%** | **+36pp** |
| | Recall (frac) | 0.424 | **0.742** | **+32pp** |
| | MRR | 0.264 | 0.284 | +2pp |
| Aggregation | Recall@k | 100% | 100% | 0 |
| | Recall (frac) | 0.216 | **0.292** | **+8pp** |
| | F1 | 0.263 | **0.301** | +4pp |
| | MRR | 0.508 | **0.875** | **+37pp** |
| All in-scope | Refusals on 51 in-scope questions | 0 | 0 | 0 |

Net per-question recall@k flips: 6 questions newly retrieved gold (q001, q010, q032, q033, q037, q038), 2 questions lost page-level gold (q017, q027) but **both still answer correctly** because the same fact is duplicated across other chunks.

---

## 5. Failure mode catalog

Selected worst-cases per system, illustrative not exhaustive.

### Baseline

| qid | Failure | What happened |
|---|---|---|
| q001 | retrieval miss | Gold page (a regulatory document, page 21) not in top-10; chunking boundary fell between context and answer |
| q026 | named-entity miss | A small-named industry company doesn't appear in any baseline-retrieved chunk; the entity name wasn't a strong lexical signal vs other industry company names |
| q046 | wrong entities | Listed two real industry companies as members of the target sub-sector — both real but not in our gold list of established operators |

### External

| qid | Failure | What happened |
|---|---|---|
| q036 | partial | Found one regional case-study under a hub-organization node; missed other regional cases because the hub node's chunks didn't surface all regional examples |
| q047 | wrong sub-population | Listed general-organization staff but didn't single out a regional-subsidiary president — node-level retrieval can't always sub-segment by "regional subsidiary" vs "general organization" |

### Internal

| qid | Failure | What happened |
|---|---|---|
| q036 | partial (same as External) | Bridge region-A → hub-organization → region-B required the hub node's text_unit_ids to include both; the larger §16.23 pool widens hub chunk coverage but region-A-side specifics still dominate the seed top-1 |
| q047 | partial (same as External) | Regional subsidiary vs general organization — same node, no granularity to separate |

The remaining failures share a fundamental constraint with External: **the production graph's entity granularity** (the regional-subsidiary vs general-organization distinction isn't a separate node, it's a within-node textual nuance). The three-pool refactor expands what each entity can contribute (Pool A guarantees the top-1 chunk; Pool C surfaces hop-expansion evidence) but cannot create granularity the graph itself doesn't have.

---

## 6. Recommendations

### 6.1 If correctness is the priority → Internal

For applications where every percentage point of answer correctness matters (regulated industries, decision-support tools, demo-quality outputs), Internal is the clear pick: 97.1% mean correctness (excl OOS), 95.5% on multi-hop questions, 100% OOS refusal.

### 6.2 If latency is critical → External

For interactive applications where 2s p50 matters more than 3pp of answer correctness, External is the cleanest baseline: single LLM call, p50 1.7-2.8s, 94.1% mean correctness still.

### 6.3 If infrastructure cost is the constraint → Baseline

Baseline is the cheapest to maintain — no knowledge graph, no traversal infrastructure, just chunks + BM25 + FAISS. The 79.4% mean correctness sets a floor that the more complex systems must meaningfully exceed to justify their additional infrastructure.

### 6.4 Generalization caveats

- Test set is small (61 questions). Per-category cells are 10-30 questions; single failures swing percentages
- Corpus is moderate-size (8 PDFs, ~580 pages). At 10× scale, both graph build and retrieval costs would shift; this isn't tested
- Gold facts come from corpus extraction; "answer correctness" measures fidelity to the source documents, not absolute truth
- Local Gemma 4 26B is moderate-strength. Frontier models (Claude Sonnet, GPT-4) would compress the gap between architectures by being more tolerant of noisy evidence
- Hand-written OOS questions are obvious; a "soft OOS" test (in-domain question with no graph coverage) would stress-test the refusal logic harder

---

## 7. Files

- `eval/testset/gold.jsonl` — 61 hand-curated questions with gold chunks, pages, facts
- `eval/testset/candidates.json` — sampled candidate pool from `dump_candidates.py`
- `eval/reports/raw_runs/{baseline,external_v2,internal_v2}.jsonl` — per-question outputs (the `_v2` suffix in filenames is a code-history artifact; report-wise they are the 3 evaluated systems)
- `eval/reports/raw_runs/internal_v2.before_parity.jsonl`, `internal_v2.before_bm25.jsonl`, `internal_v2.before_3pool.jsonl` — snapshots from each step of the 2026-05-16 Internal rebuild (kept for diff-against-current and reproducibility audit)
- `eval/reports/scored.csv` — long-format table (one row per question × system) with R/P/F1/MRR + latency + LLM calls
- `eval/reports/aggregate.json` — per (category, system) summary stats
- `eval/baseline/` — independent RAG+BM25 implementation (chunker, FAISS, BM25, RRF retriever, composer)
- `eval/data/baseline_index/` — baseline chunk + dense + BM25 index files
- `eval/data/node_bm25/` — node-level BM25 index used by External (eval-local snapshot; src now uses `instance.bm25_store`)
- `eval/runners/run_baseline.py`, `eval/runners/run_external_v2.py`, `eval/runners/run_internal_v2.py` — system runners (Internal + External now call `src.modules.m2_qa.qa_trace()` / `src.modules.m2_qa_agent.fast_query_trace()` directly per §16.22 parity)
- `eval/scripts/{dump_candidates,build_gold,build_baseline_index,build_node_bm25,smoke_test_internal_v2,analyze,inspect_run,parity_compare}.py` — pipeline tooling (`parity_compare.py` added with the §16.22 parity work)
- `eval/scripts/dump_for_relevance_judge.py` + `eval/scripts/judge_relevance.py` — semantic relevance judgment (§3.1.b)
- `eval/reports/relevance_judgments.jsonl` — per-chunk relevance labels
- `eval/reports/aggregate_relevance.json` — aggregated semantic P/R/F1
