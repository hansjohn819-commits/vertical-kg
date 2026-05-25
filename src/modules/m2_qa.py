"""M2 Q&A pipeline — deterministic GraphRAG-style retrieval + composer.

Replaces the old agent-loop-based GraphAgent.call() flow for user questions.
Architecture (validated end-to-end in eval/ at 97% answer correctness):

  question
    │
    ▼
  [1] LLM decompose → 1-5 sub-questions (thinking=off, ~2-3s)
    │
    ▼
  [2] Per sub-q: mechanical k-hop frontier traversal — NO LLM in loop
        seeds = dense FAISS top-8 + label-substring-match (up to 5 extra)
        for hop in 1..3:
          candidates = frontier's neighbors (deduped)
          score = cosine(sub_q, neighbor) × edge_type_weight
          keep top-8 above FRONTIER_THRESHOLD=0.4
          early-stop on frontier collapse
        collect visited nodes + their chunks
    │
    ▼
  [3] Aggregate across sub-qs: dedupe chunks + edges (both endpoints in visited)
    │
    ▼
  [4] OOS pre-filter: max_seed_score < 0.30 → canned refusal, skip composer
    │
    ▼
  [5] Build evidence:
        - GRAPH RELATIONSHIPS section (first; ranked by combined endpoint degree)
        - EVIDENCE PASSAGES section (per-sub-q chunk rerank + force-include
          hop-0 seed chunks; cap 20 chunks within 24K token budget)
    │
    ▼
  [6] Composer LLM call (thinking=off, step-by-step prompt for cross-reference)
    │
    ▼
  [7] OOS post-filter: refusal-shape + no doc cited + borderline seed score
        → rewrite to canned refusal
    │
    ▼
  return answer

Why thinking=off: Gemma 4 + current llama.cpp build (b9106) wedges on thinking-on
generation (llama.cpp Discussion #21338). Cross-reference reasoning that
thinking-on would do is instead encoded as explicit step-by-step instructions
in COMPOSER_SYSTEM_PROMPT.

Seed retrieval uses hybrid dense+BM25 (RRF) over node summaries, the same
``instance.bm25_store`` external ``fast_query`` uses. The original v2 build
ran dense-only at the seed step ("path B" in the eval iteration), with
label-substring matching as the lexical fallback. §16.23 reproduction
exposed cases where lexically-distinctive entities ("Argyle Aquaculture
Development Area" for a "Licensed Aquaculture Sites" question) never
appeared in dense top-20 yet ranked top-10 under RRF fusion; label-match
couldn't promote them either (low specificity vs the 30+ other
"Aquaculture"-containing entities). Switching the seed step to hybrid
restores parity with the external retrieval design.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from src.graph.instance import GraphInstance
from src.graph.retrieval import (
    _get_model, _node_text, display_title, encode_node, encode_query, top_k,
)
from src.graph.tokens import count_tokens
from src.graph.traversal_log import append_query
from src.llm.local_client import LocalClient
from src.llm.routing import get_client


# ---------------------------------------------------------------------------
# Tunable parameters (validated by eval — see eval/reports/summary.md §9)
# ---------------------------------------------------------------------------

MAX_SUBQUESTIONS = 4
HOP_BUDGET = 3
BEAM_WIDTH = 8
# Seed-step retrieval pool size, decoupled from BEAM_WIDTH so the hop
# expansion budget can stay tight while seeds match external fast_query's
# k=10. RRF fusion sometimes places the actually-relevant entity at rank
# 9-10 (§16.23 "Argyle Aquaculture Development Area" example); BEAM_WIDTH=8
# would clip it before traversal even starts.
SEED_K = 10
FRONTIER_THRESHOLD = 0.40
OOS_THRESHOLD = 0.30                # seed top-1 cosine — below this = OOS
LABEL_MATCH_MAX_EXTRA = 5
LABEL_MATCH_MIN_TOKEN_LEN = 3
EDGES_BUDGET_TOKENS = 2000
EDGES_TOP_N = 60

# §16.23 three-pool chunk selection
#   Pool A (RRF seeds): each dense+BM25 seed contributes its top-1 cosine
#     chunk. Across sub-qs, deduped by node_id (winner = max-cosine sub-q).
#     Cap RRF_SEED_CHUNK_CAP. If unique seeds exceed cap, the chunks are
#     re-ranked against the original question and top-N kept.
#   Pool B (label-match seeds): same idea but for label-substring seeds.
#     Cap LABEL_SEED_CHUNK_CAP.
#   Pool C (rerank): everything else (seeds' non-top-1 chunks + hop-expanded
#     visited chunks) goes through a per-sub-q rerank that fuses dense
#     cosine + ad-hoc chunk-level BM25 via RRF. Cap RERANK_CHUNK_CAP.
# Total chunks in prompt ≤ RRF_SEED_CHUNK_CAP + LABEL_SEED_CHUNK_CAP +
# RERANK_CHUNK_CAP = 35.
RRF_SEED_CHUNK_CAP = 20
LABEL_SEED_CHUNK_CAP = 5
RERANK_CHUNK_CAP = 10
# Bumped from 24K to match the relaxed prompt budget (backend has 128K
# context; empirical safe-zone for instruction-following is ~60K).
CHUNKS_BUDGET_TOKENS = 60000
# Chunk-level BM25 RRF constant — matches src.graph.retrieval.RRF_K (60)
# convention so the fusion behaviour is consistent across seed-step and
# chunk-step.
CHUNK_RRF_K = 60

# Edge-type weights for frontier scoring. Strong-semantic relations (causal,
# role, authorship) > medium (attribute / membership) > weak (RELATED_TO).
EDGE_WEIGHTS = {
    # Strong-semantic
    "AUTHORED_WITH": 1.5, "CO_AUTHORED": 1.5, "CO_FOUNDED": 1.5,
    "CEO_OF": 1.5, "FOUNDED": 1.5, "AUTHORED_BY": 1.5,
    "PUBLISHED_BY": 1.5, "PUBLISHED": 1.5, "OWNED_BY": 1.5,
    "AUDITED_BY": 1.5, "AUDITS": 1.5, "REGULATES": 1.5,
    "FUNDED": 1.5, "FUNDED_BY": 1.5, "PARTNERED_WITH": 1.5,
    "PARTNERS_IN": 1.5, "ACQUIRED": 1.5, "MERGED_WITH": 1.5,
    # Medium-semantic
    "AFFILIATED_WITH": 1.2, "IN_INDUSTRY": 1.2, "MEMBERSHIP_OF": 1.2,
    "MEMBERS_OF": 1.2, "PRODUCES": 1.2, "PRODUCED_BY": 1.2,
    "PRODUCED_IN": 1.2, "OPERATES_IN": 1.2, "LOCATED_IN": 1.2,
    "HEADQUARTERED_IN": 1.2, "EXPORTS": 1.2, "IMPORTED_FROM": 1.2,
    "IMPORTS": 1.2, "STUDIED_LOCATION": 1.2, "CONTAINS": 1.2,
    "INCLUDES": 1.2, "PART_OF": 1.2,
    # Weak
    "RELATED_TO": 0.7,
}
EDGE_WEIGHT_DEFAULT = 1.0


LABEL_MATCH_STOPWORDS = {
    "Who", "What", "Which", "Where", "When", "Why", "How", "Did",
    "Has", "Have", "Is", "Are", "Was", "Were", "Do", "Does",
    "Can", "Could", "Will", "Would", "Should", "May", "Might",
    "The", "A", "An", "And", "Or", "But", "Not", "If", "Yes", "No",
    "Common", "Same", "Both", "Several", "List", "Name",
}
_CAP_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z'.\-]*\b")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DECOMPOSER_SYSTEM_PROMPT = """You are a query decomposer for a knowledge-graph retrieval system over a corpus of kelp / seaweed industry reports.

Given a user question, output 1-5 atomic sub-questions that together fully cover the original. Each sub-question must:
- Focus on a single named entity, relationship, or fact
- Be self-contained (no pronouns referring to other sub-questions)
- Be a noun phrase or full question

Rules:
- If the question is atomic (single fact lookup), output ONLY the original question verbatim
- For multi-hop bridge questions, decompose by entity (one sub-q per endpoint, plus optionally one for the bridge)
- For aggregation questions, output 2-3 phrasings to widen recall
- For out-of-scope questions (Bitcoin price, restaurant recommendations, weather, etc.), output the original verbatim — do NOT try to translate or reframe
- Output ONE sub-question per line. NO numbering, NO preamble, NO explanation, NO blank lines.

Examples:

Q: Who is the CEO of Atlantic Sea Farms?
SUB:
Who is the CEO of Atlantic Sea Farms?

Q: Are R. Fujita and M. Stekoll connected as co-authors through a common collaborator?
SUB:
Who has R. Fujita co-authored with?
Who has M. Stekoll co-authored with?
Common co-authors of R. Fujita and M. Stekoll

Q: Name companies operating in the North American kelp industry.
SUB:
Companies in the North American kelp industry
Kelp companies operating in North America
North American seaweed industry players

Q: What is the capital of Mongolia?
SUB:
What is the capital of Mongolia?
"""


COMPOSER_SYSTEM_PROMPT = """You are a kelp / seaweed industry analyst. Answer using ONLY the evidence below.

The evidence has TWO parts (RELATIONSHIPS first, then PASSAGES):
1. GRAPH RELATIONSHIPS — verified edges between entities, format "<A> -[<rel>]-> <B>" plus an optional verbatim evidence quote. Each line is an established fact.
2. EVIDENCE PASSAGES — verbatim source text from named reports, ranked by relevance.

# How to answer different question shapes

## (a) "Are A and B connected through a common X?" / "Common collaborator/link/bridge between A and B?" / "Through whom are A and B connected?"

Do this mechanically before deciding to refuse:
  Step 1. From the RELATIONSHIPS list, list every entity X where you see an edge involving A and X (in either direction). Call this set S_A.
  Step 2. Same for B → S_B.
  Step 3. Compute the intersection S_A ∩ S_B. Every entity in the intersection is a verified common link.
  Step 4. If intersection is non-empty, answer: "Yes, [list the common entities] is/are the common [collaborator/link]."
  Step 5. ONLY if the intersection is genuinely empty, say no common link is documented.

DO NOT default to "I don't have information" before performing steps 1-3 above. The relationships list is sufficient — you do NOT need a single passage that mentions all parties together.

## (b) "Where/in which X is Y based/located/active?" / single-fact lookups

Look for the relevant edge in RELATIONSHIPS first (HEADQUARTERED_IN, LOCATED_IN, OPERATES_IN, AFFILIATED_WITH, etc.), then cross-check in PASSAGES.

## (c) "List/name several X" (aggregation)

Scan both RELATIONSHIPS (for edges of the requested type) and PASSAGES, dedupe entities, list them with one-line context each.

## (d) Yes/no questions

Same: check RELATIONSHIPS for the relevant edge first.

# Strict grounding (always)

- Answer ONLY from this evidence. Do not add facts from prior training (capitals, sports scores, code, recipes, translations, weather, current events).
- If both RELATIONSHIPS and PASSAGES truly lack relevant information, plainly say "I don't have information on that" and stop. But always perform the appropriate scan first.
- Do not speculate. Do not pad with related-but-different facts.

# Style

- Cite sources by report title in prose (e.g., "according to the State of the Kelp Industry Report"). Drop ".pdf" and similar suffixes.
- DO NOT mention the structure of the evidence: never write "according to the GRAPH RELATIONSHIPS section" or "in the EVIDENCE PASSAGES". Describe sources by report title or speak in your own analyst voice.
- Be concise — direct prose, not a summary of the evidence list.
"""


REFUSAL_TEMPLATE = "I don't have information on that in my industry coverage. My expertise is the kelp / seaweed industry as represented in the corpus."


# ---------------------------------------------------------------------------
# Step 1: decompose
# ---------------------------------------------------------------------------

def _decompose_query(client: LocalClient, question: str) -> list[str]:
    """1 LLM call (thinking=off). Falls back to [question] on any failure."""
    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": DECOMPOSER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Q: {question}\nSUB:"},
            ],
            temperature=0.3,
            thinking=False,
            timeout=30,
        )
        out = (resp.choices[0].message.content or "").strip()
    except Exception:
        return [question]

    lines = [
        ln.strip().lstrip("-*•").strip()
        for ln in out.split("\n")
        if ln.strip()
    ]
    lines = [
        ln for ln in lines
        if not ln.lower().startswith(("sub:", "sub-questions", "subquestions", "answer:", "q:"))
        and len(ln) >= 5
    ]
    if not lines:
        return [question]
    seen = set()
    out_subs: list[str] = []
    for ln in lines:
        key = ln.lower()
        if key in seen:
            continue
        seen.add(key)
        out_subs.append(ln)
        if len(out_subs) >= MAX_SUBQUESTIONS:
            break
    return out_subs or [question]


# ---------------------------------------------------------------------------
# Step 2: mechanical traversal (per sub-q)
# ---------------------------------------------------------------------------

@dataclass
class TraversalResult:
    sub_q: str
    visited_node_ids: set = field(default_factory=set)
    chunk_ids: set = field(default_factory=set)
    # Per-seed top-1 chunks (selected by cosine of chunk text vs sub_q).
    # Two pools matching the §16.23 design: RRF (dense+BM25 fusion) seeds
    # and label-substring-match seeds. Value is (chunk_id, cosine_score).
    rrf_seed_top_chunks: dict = field(default_factory=dict)
    label_seed_top_chunks: dict = field(default_factory=dict)
    seed_scores: list = field(default_factory=list)
    hops_used: int = 0


def _label_match_seeds(storage, sub_q: str) -> list:
    """Token-level label substring match. Disambiguation fallback for entity
    names that dense FAISS misses (e.g., bibliography-style "Last, First").
    Not a parallel retrieval system — just ensures user-mentioned entities
    are in seeds."""
    raw_tokens = _CAP_TOKEN_RE.findall(sub_q or "")
    tokens = [
        t for t in raw_tokens
        if t not in LABEL_MATCH_STOPWORDS
        and len(t) >= LABEL_MATCH_MIN_TOKEN_LEN
    ]
    if not tokens:
        return []
    tokens_lower = [t.lower() for t in tokens]

    matches: list[tuple[int, float, object]] = []
    for n in storage.nodes():
        if n.merged_into is not None:
            continue
        label_l = (n.label or "").lower()
        if not label_l:
            continue
        hit = sum(1 for t in tokens_lower if t in label_l)
        if hit == 0:
            continue
        n_words = max(1, len(n.label.split()))
        specificity = hit / n_words
        matches.append((hit, specificity, n))
    matches.sort(key=lambda x: (-x[0], -x[1]))
    return [m[2] for m in matches[:LABEL_MATCH_MAX_EXTRA]]


def _mechanical_traversal(instance: GraphInstance, sub_q: str) -> TraversalResult:
    storage = instance.storage
    result = TraversalResult(sub_q=sub_q)

    sub_q_emb = encode_query(sub_q)

    # Seeds: hybrid dense+BM25 (RRF) over node summaries + label-match
    # disambiguation. BM25 was originally absent here (path B in the
    # eval iteration), but the §16.23 reproduction showed that purely
    # dense top-N misses lexically-distinctive entities even when they
    # exist in the graph — e.g. "Licensed Aquaculture Sites" query never
    # surfaced "Argyle Aquaculture Development Area" within dense top-20,
    # while RRF fusion put it at rank 9. Reusing instance.bm25_store
    # (same index external fast_query uses) costs no new infrastructure.
    dense_seeds = top_k(
        instance.vector_store, storage, sub_q, k=SEED_K,
        bm25_store=instance.bm25_store,
    )
    label_seeds = _label_match_seeds(storage, sub_q)
    dense_seed_ids = {s.id for s in dense_seeds}
    label_match_only_ids: set = set()
    seeds = list(dense_seeds)
    for n in label_seeds:
        if n.id not in dense_seed_ids:
            seeds.append(n)
            label_match_only_ids.add(n.id)

    seeds = seeds[: SEED_K + LABEL_MATCH_MAX_EXTRA]
    if not seeds:
        return result

    # Seed scoring (cosine sub_q vs node) — batch encode
    model = _get_model()
    seed_texts = [_node_text(s) for s in seeds]
    seed_vecs = model.encode(
        seed_texts, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    )
    seed_cos = (seed_vecs @ sub_q_emb).tolist()
    result.seed_scores = seed_cos

    for s in seeds:
        result.visited_node_ids.add(s.id)
        result.chunk_ids.update(s.text_unit_ids or [])

    # Per-seed top-1 chunk (§16.23 pool A/B). Each seed contributes its
    # chunk with the highest cosine vs sub_q. Routed to rrf_seed_top_chunks
    # for RRF/BM25-fused seeds or label_seed_top_chunks for label-match-
    # only seeds. _build_evidence treats them as two separate pools with
    # independent caps.
    text_units = instance.text_units
    for s in seeds:
        if not s.text_unit_ids:
            continue
        chunks = [(cid, text_units.get(cid)) for cid in s.text_unit_ids]
        chunks = [(cid, cd) for cid, cd in chunks if cd is not None]
        if not chunks:
            continue
        if len(chunks) == 1:
            top_cid, top_cos = chunks[0][0], 1.0
        else:
            texts = [cd.get("text", "") or "" for _, cd in chunks]
            vecs = model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            )
            scores = (vecs @ sub_q_emb).tolist()
            best_i = max(range(len(chunks)), key=lambda i: scores[i])
            top_cid, top_cos = chunks[best_i][0], float(scores[best_i])
        if s.id in label_match_only_ids:
            result.label_seed_top_chunks[s.id] = (top_cid, top_cos)
        else:
            result.rrf_seed_top_chunks[s.id] = (top_cid, top_cos)

    frontier = list(seeds)

    for hop in range(HOP_BUDGET):
        # Collect candidate neighbors (deduped per nbr_id, keep best edge_boost)
        candidate_nbrs: dict[str, tuple] = {}
        for n in frontier:
            for nbr in storage.neighbors(n.id):
                if nbr.id in result.visited_node_ids:
                    continue
                if nbr.merged_into is not None:
                    continue
                edge_types = storage.edge_types_between(n.id, nbr.id)
                if not edge_types:
                    continue
                max_boost = max(
                    EDGE_WEIGHTS.get(t, EDGE_WEIGHT_DEFAULT) for t in edge_types
                )
                if nbr.id not in candidate_nbrs or candidate_nbrs[nbr.id][3] < max_boost:
                    candidate_nbrs[nbr.id] = (n, nbr, edge_types, max_boost)

        if not candidate_nbrs:
            break

        candidates = list(candidate_nbrs.values())
        nbr_texts = [_node_text(c[1]) for c in candidates]
        nbr_vecs = model.encode(
            nbr_texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        scored = []
        for i, (parent, nbr, edge_types, edge_boost) in enumerate(candidates):
            cos = float(nbr_vecs[i] @ sub_q_emb)
            score = cos * edge_boost
            scored.append((score, parent, nbr, edge_types))
        scored.sort(key=lambda x: -x[0])

        # Frontier collapse — top score below threshold means nothing useful
        if scored[0][0] < FRONTIER_THRESHOLD:
            break

        next_frontier = []
        for score, parent, nbr, edge_types in scored[:BEAM_WIDTH]:
            if score < FRONTIER_THRESHOLD:
                break
            result.visited_node_ids.add(nbr.id)
            result.chunk_ids.update(nbr.text_unit_ids or [])
            next_frontier.append(nbr)

        if not next_frontier:
            break

        result.hops_used = hop + 1
        frontier = next_frontier

    return result


# ---------------------------------------------------------------------------
# Step 3: aggregate
# ---------------------------------------------------------------------------

@dataclass
class Aggregated:
    chunk_ids: set
    visited_node_ids: set
    edges_payload: list
    max_seed_score: float
    per_sub_traces: list
    # Per-seed top-1 chunks deduped across sub-qs.  When the same node
    # appears as a seed in multiple sub-qs, the entry with the highest
    # cosine vs its sub_q wins (matches user spec "用 cosine 最高").  Values
    # are (chunk_id, cosine_score).
    rrf_seed_top_chunks: dict = field(default_factory=dict)
    label_seed_top_chunks: dict = field(default_factory=dict)


def _collect_edges_between_visited(instance: GraphInstance, visited: set) -> list[dict]:
    storage = instance.storage
    seen: set[str] = set()
    out: list[dict] = []
    for nid in visited:
        for e in storage.incident_edges(nid):
            if e.id in seen:
                continue
            if e.source_id not in visited or e.target_id not in visited:
                continue
            src = storage.get_node(e.source_id)
            tgt = storage.get_node(e.target_id)
            if src is None or tgt is None:
                continue
            seen.add(e.id)
            out.append({
                "edge_id": e.id,
                "src_label": src.label, "src_type": src.type,
                "tgt_label": tgt.label, "tgt_type": tgt.type,
                "type": e.type,
                "evidence_quote": e.evidence_quote,
                "hub_score": storage.degree(e.source_id) + storage.degree(e.target_id),
            })
    return out


def _aggregate(instance: GraphInstance, traversals: list[TraversalResult]) -> Aggregated:
    chunk_ids: set = set()
    visited: set = set()
    rrf_top: dict = {}      # node_id -> (chunk_id, cos)
    label_top: dict = {}
    max_seed = 0.0
    for t in traversals:
        chunk_ids |= t.chunk_ids
        visited |= t.visited_node_ids
        # Dedup by node_id, keep entry with highest cosine across sub-qs.
        for nid, payload in t.rrf_seed_top_chunks.items():
            if nid not in rrf_top or payload[1] > rrf_top[nid][1]:
                rrf_top[nid] = payload
        for nid, payload in t.label_seed_top_chunks.items():
            if nid not in label_top or payload[1] > label_top[nid][1]:
                label_top[nid] = payload
        if t.seed_scores:
            max_seed = max(max_seed, max(t.seed_scores))
    edges = _collect_edges_between_visited(instance, visited)
    return Aggregated(
        chunk_ids=chunk_ids,
        visited_node_ids=visited,
        edges_payload=edges,
        max_seed_score=max_seed,
        per_sub_traces=traversals,
        rrf_seed_top_chunks=rrf_top,
        label_seed_top_chunks=label_top,
    )


# ---------------------------------------------------------------------------
# Step 5: build evidence
# ---------------------------------------------------------------------------

def _build_evidence(instance: GraphInstance, agg: Aggregated, original_q: str,
                    traversals: list[TraversalResult]) -> tuple[str, list[str], list[str]]:
    """Three-pool §16.23 chunk selection:

      Pool A — RRF-seed top-1 chunks (cap RRF_SEED_CHUNK_CAP=20).  When unique
        seeds across sub-qs exceed the cap, the top-1 chunks themselves are
        re-ranked against the *original* question and the top RRF_SEED_CHUNK_CAP
        kept.
      Pool B — Label-match-seed top-1 chunks (cap LABEL_SEED_CHUNK_CAP=5).
      Pool C — Hop-expanded + seeds' non-top-1 chunks, fused dense+ad-hoc-BM25
        per sub-q, take top RERANK_CHUNK_CAP=10 globally.

    Order in the prompt: Pool A → Pool B → Pool C (highest-confidence
    "entity says this" evidence first, broader rerank pool after).
    """
    text_units = instance.text_units
    chunks_by_id: dict[str, dict] = {}
    for cid in agg.chunk_ids:
        cd = text_units.get(cid)
        if cd is not None:
            chunks_by_id[cid] = cd

    if not chunks_by_id and not agg.edges_payload:
        return "(no evidence in working memory yet)", [], []

    selected_chunk_ids: list[str] = []
    chunk_blocks: list[str] = []
    used_tokens = 0

    def _render_chunk(cid: str) -> tuple[str, int]:
        cd = chunks_by_id[cid]
        rid = cd.get("raw_doc_id", "") or ""
        page = cd.get("page_num")
        title = display_title(rid)
        header = (
            f'  [from "{title}", page {page}]'
            if page is not None else f'  [from "{title}"]'
        )
        body = "  " + (cd.get("text", "") or "").strip()
        rendered = header + "\n" + body
        return rendered, count_tokens(rendered)

    def _append(cid: str) -> bool:
        nonlocal used_tokens
        if cid in selected_chunk_ids or cid not in chunks_by_id:
            return False
        rendered, ct = _render_chunk(cid)
        if used_tokens + ct > CHUNKS_BUDGET_TOKENS:
            return False
        chunk_blocks.append(rendered)
        selected_chunk_ids.append(cid)
        used_tokens += ct
        return True

    model = _get_model()

    # ---- Pool A: RRF seed top-1 chunks ----
    # dict[node_id] -> (chunk_id, cosine_vs_sub_q)
    rrf_pool: list[tuple[str, float]] = [
        (cid, cos) for (cid, cos) in agg.rrf_seed_top_chunks.values()
    ]
    # Dedup by chunk_id (different nodes can share a chunk). Keep the
    # max-cosine record so the overflow rerank later is well-defined.
    rrf_by_chunk: dict[str, float] = {}
    for cid, cos in rrf_pool:
        if cid not in rrf_by_chunk or cos > rrf_by_chunk[cid]:
            rrf_by_chunk[cid] = cos
    if len(rrf_by_chunk) > RRF_SEED_CHUNK_CAP:
        # Re-rank top-1s against the ORIGINAL question (cross-sub-q tiebreaker)
        # and keep the top RRF_SEED_CHUNK_CAP.
        cand_cids = list(rrf_by_chunk.keys())
        qv = encode_query(original_q)
        texts = [chunks_by_id[c].get("text", "") for c in cand_cids if c in chunks_by_id]
        cids_present = [c for c in cand_cids if c in chunks_by_id]
        if texts:
            vecs = model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            )
            scores = (vecs @ qv).tolist()
            ranked = sorted(zip(cids_present, scores), key=lambda x: -x[1])
            rrf_ordered = [c for c, _ in ranked[:RRF_SEED_CHUNK_CAP]]
        else:
            rrf_ordered = cids_present[:RRF_SEED_CHUNK_CAP]
    else:
        rrf_ordered = sorted(rrf_by_chunk.keys(), key=lambda c: -rrf_by_chunk[c])
    for cid in rrf_ordered:
        _append(cid)

    # ---- Pool B: label-match seed top-1 chunks ----
    label_by_chunk: dict[str, float] = {}
    for cid, cos in agg.label_seed_top_chunks.values():
        if cid not in label_by_chunk or cos > label_by_chunk[cid]:
            label_by_chunk[cid] = cos
    label_ordered = sorted(label_by_chunk.keys(), key=lambda c: -label_by_chunk[c])
    n_label_added = 0
    for cid in label_ordered:
        if n_label_added >= LABEL_SEED_CHUNK_CAP:
            break
        if _append(cid):
            n_label_added += 1

    # ---- Pool C: rerank of remaining chunks (dense cos + ad-hoc BM25 via RRF) ----
    pool_a_b = set(selected_chunk_ids)
    if chunks_by_id:
        per_sub_chunks: list[tuple[str, list[str]]] = []
        seen_in_groups: set[str] = set()
        for t in traversals:
            this_sub_chunks = [
                cid for cid in t.chunk_ids
                if cid in chunks_by_id
                and cid not in pool_a_b
                and cid not in seen_in_groups
            ]
            if this_sub_chunks:
                per_sub_chunks.append((t.sub_q, this_sub_chunks))
                seen_in_groups.update(this_sub_chunks)

        rrf_pool_scores: dict[str, float] = {}
        for sub_q, cids in per_sub_chunks:
            if not cids:
                continue
            # Dense cosine ranking.
            sub_qv = encode_query(sub_q)
            chunk_texts = [chunks_by_id[c].get("text", "") for c in cids]
            chunk_vecs = model.encode(
                chunk_texts, convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            )
            dense_scores = (chunk_vecs @ sub_qv).tolist()
            dense_ranked = sorted(
                zip(cids, dense_scores), key=lambda x: -x[1],
            )
            dense_rank = {cid: i for i, (cid, _) in enumerate(dense_ranked)}

            # Ad-hoc chunk-level BM25 over the SAME pool (no persistent index).
            # rank_bm25 is already a dep (used by node-level BM25Store).
            from rank_bm25 import BM25Okapi
            from src.graph.bm25_store import _tokenize
            tokenized_pool = [_tokenize(t) for t in chunk_texts]
            if any(tokenized_pool):
                bm25 = BM25Okapi(tokenized_pool)
                bm25_scores = bm25.get_scores(_tokenize(sub_q)).tolist()
                bm25_ranked = sorted(
                    zip(cids, bm25_scores), key=lambda x: -x[1],
                )
                bm25_rank = {cid: i for i, (cid, _) in enumerate(bm25_ranked)}
            else:
                bm25_rank = {cid: i for i, cid in enumerate(cids)}

            # RRF fuse the two rank lists per sub-q.
            for cid in cids:
                rrf = (
                    1.0 / (CHUNK_RRF_K + dense_rank[cid] + 1)
                    + 1.0 / (CHUNK_RRF_K + bm25_rank[cid] + 1)
                )
                if cid not in rrf_pool_scores or rrf > rrf_pool_scores[cid]:
                    rrf_pool_scores[cid] = rrf

        rerank_ordered = sorted(
            rrf_pool_scores.keys(), key=lambda c: -rrf_pool_scores[c],
        )
        n_rerank_added = 0
        for cid in rerank_ordered:
            if n_rerank_added >= RERANK_CHUNK_CAP:
                break
            if _append(cid):
                n_rerank_added += 1

    # Edges sorted by hub_score (prominence), capped by EDGES_TOP_N + budget
    edge_lines: list[str] = []
    edge_used = 0
    for e in sorted(agg.edges_payload, key=lambda x: -x["hub_score"])[:EDGES_TOP_N]:
        line = f"- {e['src_label']} -[{e['type']}]-> {e['tgt_label']}"
        if e["evidence_quote"]:
            quote = e["evidence_quote"][:200].replace("\n", " ").strip()
            line += f'\n  evidence: "{quote}"'
        ct = count_tokens(line)
        if edge_used + ct > EDGES_BUDGET_TOKENS:
            break
        edge_lines.append(line)
        edge_used += ct

    sections: list[str] = []
    if edge_lines:
        sections.append(
            "GRAPH RELATIONSHIPS (authoritative for connection / link questions; "
            "each line is a verified edge between entities, sorted by hub prominence):\n"
            + "\n".join(edge_lines)
        )
    if chunk_blocks:
        sections.append(
            "EVIDENCE PASSAGES (verbatim source text, ranked by relevance):\n\n"
            + "\n\n".join(chunk_blocks)
        )

    selected_doc_titles = sorted({
        display_title(chunks_by_id[c].get("raw_doc_id", "")) for c in selected_chunk_ids
        if c in chunks_by_id
    })
    return "\n\n".join(sections), selected_chunk_ids, selected_doc_titles


# ---------------------------------------------------------------------------
# Step 4: OOS pre-filter
# ---------------------------------------------------------------------------
# (The OOS post-filter was retired 2026-05-16: the §16.21 deterministic
# pipeline + §16.23 three-pool design eliminate the prompt-regression /
# fabricated-tool-result attack surface that motivated it; remaining
# refusal correctness comes from the composer prompt's grounding rules
# plus the pre-filter. Post-filter became a UX bug — when it rewrote a
# streamed answer it produced a confusing two-answer-with-rule render.)

def _oos_prefilter(agg: Aggregated) -> bool:
    """Return True if this is OOS — skip composer entirely."""
    return agg.max_seed_score < OOS_THRESHOLD


# ---------------------------------------------------------------------------
# Step 6: composer
# ---------------------------------------------------------------------------

# §16.1 / §12.5 token guard. Backend has 128K context. Evidence may take
# up to CHUNKS_BUDGET_TOKENS = 60K, generation output + system prompt +
# safety overhead reserve ~10K, leaving ~58K for (history + question).
# Past this the long-running conversation triggers an LLM auto-summary
# that collapses the transcript into one synthetic system message.
HISTORY_QUESTION_BUDGET_TOKENS = 58000

SUMMARIZE_HISTORY_SYSTEM_PROMPT = """You are a conversation compressor.

You will receive a transcript of an earlier conversation between a user
and a knowledge-graph analyst assistant. Your job is to produce a
concise summary that preserves everything the next turn might need:

- The user's overall topic and what they are exploring.
- Specific entities, numbers, names, places, or dates the user pinned
  or the assistant cited.
- Facts the assistant already stated, especially answers to past
  questions the user might refer back to with pronouns.
- Open threads — questions the user asked that were not fully
  answered, or points they pushed back on.

Drop pleasantries, restate-of-questions, scaffolding, and anything that
won't help interpret the next message. Be terse. No headings, no bullet
labels — one continuous compact paragraph. Aim for under 600 words but
do not pad to reach it.
"""


def _maybe_summarize_history(client: LocalClient,
                             history: list[dict] | None,
                             question: str,
                             *,
                             budget: int = HISTORY_QUESTION_BUDGET_TOKENS,
                             ) -> list[dict] | None:
    """Return ``history`` unchanged if it fits the budget, otherwise
    collapse it into one summary system-message via an LLM call.

    The summary call uses ``thinking=False`` — compression is a
    text-shape task, not a reasoning task — and a low temperature so
    repeat calls on the same history stay stable. If the summarization
    call itself fails for any reason, we degrade to the trivial fix of
    dropping ``history`` entirely; the alternative (raising) would
    crash the whole QA path and the user just asked one question.
    """
    if not history:
        return history
    h_tokens = sum(count_tokens(m.get("content", "") or "") for m in history)
    q_tokens = count_tokens(question or "")
    if h_tokens + q_tokens <= budget:
        return history

    transcript = "\n".join(
        f"{(m.get('role') or 'user').upper()}: {m.get('content') or ''}"
        for m in history
    )
    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": SUMMARIZE_HISTORY_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            temperature=0.3, thinking=False,
        )
        summary = (resp.choices[0].message.content or "").strip()
    except Exception:
        summary = ""
    if not summary:
        # Fall back to dropping history rather than crashing the whole
        # composer call. The current question still has full evidence;
        # losing transcript context is a downgrade, not a failure.
        return None
    return [{
        "role": "system",
        "content": f"Earlier conversation summary:\n{summary}",
    }]


def _composer_messages(evidence_text: str, question: str,
                       history: list[dict] | None = None) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": COMPOSER_SYSTEM_PROMPT}]
    if history:
        msgs.extend(history)
    msgs.append({
        "role": "user",
        "content": f"EVIDENCE:\n{evidence_text}\n\nQUESTION: {question}",
    })
    return msgs


def _compose(client: LocalClient, evidence_text: str, question: str,
             history: list[dict] | None = None,
             timeout: float = 120) -> str:
    """Composer LLM call, thinking=off. max_retries=0 to fail fast on the
    rare llama.cpp wedge (vs SDK default of 2 retries × 120s = 360s)."""
    no_retry_client = client._client.with_options(max_retries=0)
    resp = no_retry_client.chat.completions.create(
        model=client.model,
        messages=_composer_messages(evidence_text, question, history),
        temperature=0.3,
        timeout=timeout,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def qa_trace(instance: GraphInstance, question: str,
             history: list[dict] | None = None,
             client: LocalClient | None = None) -> dict:
    """Run the full Q&A pipeline and return a trace dict.

    Used by:
      - ``qa()`` — returns just the answer string
      - eval runners — need the full retrieval trace for unified scoring

    Returned keys:
      answer (str), sub_questions (list[str]), selected_chunk_ids (list[str]),
      selected_doc_titles (list[str]), max_seed_score (float),
      n_visited (int), n_chunks_collected (int), n_edges_collected (int),
      oos_prefilter_triggered (bool), oos_postfilter_triggered (bool),
      raw_answer_before_postfilter (str | None), llm_calls (int),
      timings_ms (dict).
    """
    client = client or get_client("backend")

    timings: dict[str, float] = {}
    t0 = time.time()
    sub_qs = _decompose_query(client, question)
    timings["decompose_ms"] = int((time.time() - t0) * 1000)

    t0 = time.time()
    traversals = [_mechanical_traversal(instance, sq) for sq in sub_qs]
    timings["traverse_ms"] = int((time.time() - t0) * 1000)

    agg = _aggregate(instance, traversals)

    if _oos_prefilter(agg):
        return {
            "answer": REFUSAL_TEMPLATE,
            "sub_questions": sub_qs,
            "selected_chunk_ids": [],
            "selected_doc_titles": [],
            "max_seed_score": agg.max_seed_score,
            "n_visited": len(agg.visited_node_ids),
            "n_chunks_collected": len(agg.chunk_ids),
            "n_edges_collected": len(agg.edges_payload),
            "oos_prefilter_triggered": True,
            "oos_postfilter_triggered": False,
            "raw_answer_before_postfilter": None,
            "llm_calls": 1,  # decomposer ran, composer skipped
            "timings_ms": timings,
        }

    t0 = time.time()
    evidence_text, selected_chunk_ids, selected_doc_titles = _build_evidence(
        instance, agg, question, traversals,
    )
    timings["build_evidence_ms"] = int((time.time() - t0) * 1000)

    # Traversal log reinforcement — bumps weights on touched nodes/edges
    # at next sleep pass. Skip on OOS pre-filter (no real visit to log).
    try:
        touched_edge_ids = [e["edge_id"] for e in agg.edges_payload]
        append_query(
            instance.storage,
            question=question,
            seed_node_ids=sorted(agg.visited_node_ids)[:10],
            touched_node_ids=sorted(agg.visited_node_ids),
            touched_edge_ids=touched_edge_ids,
        )
    except Exception:
        pass  # logging is best-effort

    # §16.1 token guard — collapse history if (history + question) would
    # blow the context budget. Happens before the composer call so the
    # actual LLM request always fits within the model window.
    history = _maybe_summarize_history(client, history, question)

    t0 = time.time()
    composer_failed = False
    try:
        raw_answer = _compose(client, evidence_text, question, history)
    except Exception:
        raw_answer = REFUSAL_TEMPLATE
        composer_failed = True
    timings["compose_ms"] = int((time.time() - t0) * 1000)

    if composer_failed:
        return {
            "answer": REFUSAL_TEMPLATE,
            "sub_questions": sub_qs,
            "selected_chunk_ids": selected_chunk_ids,
            "selected_doc_titles": selected_doc_titles,
            "max_seed_score": agg.max_seed_score,
            "n_visited": len(agg.visited_node_ids),
            "n_chunks_collected": len(agg.chunk_ids),
            "n_edges_collected": len(agg.edges_payload),
            "oos_prefilter_triggered": False,
            "oos_postfilter_triggered": False,
            "raw_answer_before_postfilter": None,
            "llm_calls": 1,  # composer errored before completing
            "timings_ms": timings,
        }

    final_answer = raw_answer if raw_answer.strip() else REFUSAL_TEMPLATE
    return {
        "answer": final_answer,
        "sub_questions": sub_qs,
        "selected_chunk_ids": selected_chunk_ids,
        "selected_doc_titles": selected_doc_titles,
        "max_seed_score": agg.max_seed_score,
        "n_visited": len(agg.visited_node_ids),
        "n_chunks_collected": len(agg.chunk_ids),
        "n_edges_collected": len(agg.edges_payload),
        "oos_prefilter_triggered": False,
        # Retained as constant False for JSONL schema compatibility; the
        # post-filter that this used to track was retired 2026-05-16.
        "oos_postfilter_triggered": False,
        "raw_answer_before_postfilter": None,
        "llm_calls": 2,
        "timings_ms": timings,
    }


def qa(instance: GraphInstance, question: str,
       history: list[dict] | None = None,
       client: LocalClient | None = None) -> str:
    """Run the full Q&A pipeline. Returns the final answer string.

    Thin wrapper around :func:`qa_trace` that drops the retrieval trace.
    """
    return qa_trace(instance, question, history=history, client=client)["answer"]


def qa_stream(instance: GraphInstance, question: str,
              history: list[dict] | None = None,
              client: LocalClient | None = None) -> Iterator[str]:
    """Streaming variant.

    Runs the deterministic prep (decompose → traverse → aggregate → OOS
    pre-filter → build evidence) blocking — none of that benefits from
    streaming — then yields composer tokens as they arrive from the LLM.
    On OOS pre-filter trigger the canned refusal is yielded as a single
    chunk and the function returns; otherwise composer tokens stream
    through one-by-one with no post-stream rewrite.
    """
    client = client or get_client("backend")

    sub_qs = _decompose_query(client, question)
    traversals = [_mechanical_traversal(instance, sq) for sq in sub_qs]
    agg = _aggregate(instance, traversals)

    if _oos_prefilter(agg):
        yield REFUSAL_TEMPLATE
        return

    evidence_text, _selected_chunk_ids, _selected_doc_titles = _build_evidence(
        instance, agg, question, traversals,
    )

    try:
        touched_edge_ids = [e["edge_id"] for e in agg.edges_payload]
        append_query(
            instance.storage,
            question=question,
            seed_node_ids=sorted(agg.visited_node_ids)[:10],
            touched_node_ids=sorted(agg.visited_node_ids),
            touched_edge_ids=touched_edge_ids,
        )
    except Exception:
        pass  # logging is best-effort

    # §16.1 token guard — same as qa_trace, before the streaming call.
    history = _maybe_summarize_history(client, history, question)

    collected: list[str] = []
    try:
        for event in client.chat_stream(
            messages=_composer_messages(evidence_text, question, history),
            temperature=0.3, thinking=False,
        ):
            if event.token:
                collected.append(event.token)
                yield event.token
    except Exception:
        # Composer wedged mid-stream — only yield the canned refusal if
        # nothing landed yet, otherwise the user keeps whatever did stream
        # (no double-message rewrite — see the retired post-filter note).
        if not collected:
            yield REFUSAL_TEMPLATE
            return
