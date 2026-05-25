"""M2 chat router + external fast-query path.

After the 2026-05-15 v2 refactor:
  - User Q&A → handled by ``src.modules.m2_qa.qa()`` (deterministic GraphRAG
    pipeline, no agent loop, no LLM-driven tool dispatch). This module's
    ``GraphAgent.call()`` delegates plain text to ``instance.qa()``.
  - Chat commands → hardcoded string routing:
        ``/ingest <filename>`` → ingest_file
        ``/ingest``            → list available files in data/raw/
        ``/sleep``             → trigger_sleep_pass
        anything else          → instance.qa()
  - External (anonymous / read-only) path → ``fast_query()`` /
    ``fast_query_stream()``, which use the hybrid dense+BM25 retrieval (§16.20)
    over node summaries plus a counter-balanced composer prompt and an OOS
    post-filter for defense in depth.

The old LLM-driven tool dispatch (graph_query / show_provenance /
list_recent_* / read_ontology / mark_stale / etc.) was removed because
empirical eval showed:
  (a) the agent leaked training-data answers on out-of-scope questions (60%
      refusal rate vs 100% for the deterministic v2 pipeline)
  (b) the multi-step tool loop frequently looped or timed out on
      aggregation queries
  (c) the tool-result-fabrication failure mode that §16.8.3 was guarding
      against is no longer reachable when there is no LLM in the dispatch
      loop.

Admin tools that were exposed via chat (show_provenance, list_recent_*,
read_ontology) remain available as Python APIs on ``GraphInstance`` /
storage but are no longer LLM-callable. If needed in the UI, add explicit
sidebar buttons or new ``/command`` strings rather than re-introducing the
agent loop.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Iterator

from src.graph.instance import GraphInstance
from src.graph.retrieval import display_title, top_k
from src.graph.tokens import (
    INGEST_SUPERCHUNK_MAX_PAGES,
    INGEST_SUPERCHUNK_OVERLAP_PAGES,
    count_tokens,
)
from src.graph.traversal_log import append_query
from src.llm.local_client import StreamEvent
from src.llm.routing import get_client
from src.modules.m2_qa import _maybe_summarize_history

# Budget for graph_query retrieval context (§12.5.2). With 128K context the
# cap is generous; limit exists to guard against pathological cases (a hub
# entity with 100 source chunks would otherwise dominate the prompt).
RETRIEVAL_BUDGET_TOKENS = 20_000
_RERANK_TOP_N_CHUNKS = 10


# ---------------------------------------------------------------------------
# Shared: chunk-evidence builder
# ---------------------------------------------------------------------------
#
# Used by the external fast_query path. Internal Q&A uses its own evidence
# builder in src.modules.m2_qa (different shape: RELATIONSHIPS section,
# per-sub-q rerank, force-included hop-0 seed chunks).

def _build_chunk_evidence(
    instance: GraphInstance,
    seeds: list,
    *,
    question: str,
    include_neighbors: bool,
    use_display_title: bool,
    top_n_chunks: int = _RERANK_TOP_N_CHUNKS,
    budget_tokens: int = RETRIEVAL_BUDGET_TOKENS,
) -> tuple[str, set[str], list[str]]:
    """Format an EVIDENCE block grouped per-entity.

    For each seed: render label + summary, then up to N top-cosine-reranked
    chunks (against ``question``) attached to that seed via text_unit_ids,
    within a shared token budget. Optional NEIGHBORS section (summary-only,
    no chunks) for the internal-agent-style legacy path.

    Returns (evidence_text, included_node_ids, selected_chunk_ids). The
    chunk_id list is in cosine-rerank-rank order — eval / provenance UIs
    need this to map back to (raw_doc_id, page_num).
    """
    from src.graph.retrieval import _get_model, encode_query, with_neighbors

    storage = instance.storage
    text_units = instance.text_units

    seed_chunk_ids: dict[str, list[str]] = {}
    chunk_payloads: dict[str, dict] = {}
    for seed in seeds:
        ids: list[str] = []
        for cid in (getattr(seed, "text_unit_ids", None) or []):
            cd = text_units.get(cid)
            if cd is None:
                continue
            ids.append(cid)
            chunk_payloads.setdefault(cid, cd)
        seed_chunk_ids[seed.id] = ids

    chunk_rank: dict[str, int] = {}
    all_chunk_ids = list(chunk_payloads.keys())
    if all_chunk_ids:
        qv = encode_query(question)
        model = _get_model()
        chunk_texts = [
            (chunk_payloads[cid].get("text", "") or "")
            for cid in all_chunk_ids
        ]
        chunk_vecs = model.encode(
            chunk_texts, convert_to_numpy=True, normalize_embeddings=True,
        )
        scores = (chunk_vecs @ qv).tolist()
        ranked = sorted(zip(all_chunk_ids, scores), key=lambda x: -x[1])
        chunk_rank = {cid: i for i, (cid, _) in enumerate(ranked)}

    selected_chunk_ids: set[str] = set()
    chunk_render_cache: dict[str, tuple[str, int]] = {}
    used_tokens = 0
    for cid, _ in sorted(chunk_rank.items(), key=lambda x: x[1]):
        if len(selected_chunk_ids) >= top_n_chunks:
            break
        cd = chunk_payloads[cid]
        rid = cd.get("raw_doc_id", "") or ""
        page = cd.get("page_num")
        title = display_title(rid) if (use_display_title and rid) else rid
        header = (
            f'  [from "{title}", page {page}]'
            if page is not None else f'  [from "{title}"]'
        )
        body = "  " + (cd.get("text", "") or "").strip()
        rendered = header + "\n" + body
        ct = count_tokens(rendered)
        if used_tokens + ct > budget_tokens:
            continue
        chunk_render_cache[cid] = (rendered, ct)
        selected_chunk_ids.add(cid)
        used_tokens += ct

    blocks: list[str] = []
    included_ids: set[str] = set()
    for seed in seeds:
        block_lines = [
            f"Entity: {seed.label} ({seed.type})",
            f"Summary: {seed.summary}",
        ]
        surviving = [
            cid for cid in seed_chunk_ids.get(seed.id, [])
            if cid in selected_chunk_ids
        ]
        surviving.sort(key=lambda cid: chunk_rank.get(cid, 1 << 30))
        if surviving:
            block_lines.append("Sources:")
            for cid in surviving:
                rendered, _ = chunk_render_cache[cid]
                block_lines.append(rendered)
        block_text = "\n".join(block_lines)
        bt = count_tokens(block_text)
        if used_tokens + bt > budget_tokens and blocks:
            break
        blocks.append(block_text)
        used_tokens += bt
        included_ids.add(seed.id)

    if include_neighbors:
        seed_ids = {s.id for s in seeds}
        neighbors = with_neighbors(storage, seeds)
        nbr_lines: list[str] = []
        for nb in neighbors:
            if nb.id in seed_ids or nb.id in included_ids:
                continue
            prov = getattr(nb, "provenance", None)
            rid = getattr(prov, "raw_doc_id", None) if prov is not None else None
            src = ""
            if rid:
                t = display_title(rid) if use_display_title else rid
                src = f" (source: {t})"
            line = f"- [{nb.type}] {nb.label}: {nb.summary}{src}"
            lt = count_tokens(line)
            if used_tokens + lt > budget_tokens:
                break
            nbr_lines.append(line)
            used_tokens += lt
            included_ids.add(nb.id)
        if nbr_lines:
            blocks.append(
                "\nNEIGHBORS (context only, no full sources):\n"
                + "\n".join(nbr_lines)
            )

    text = "\n\n".join(blocks).strip()
    # Selected chunk_ids in rerank-rank order (drop any that didn't survive
    # the per-seed Sources filtering — those weren't actually rendered).
    selected_chunk_ids = sorted(
        selected_chunk_ids, key=lambda cid: chunk_rank.get(cid, 1 << 30),
    )
    return (
        text or "(no evidence in working memory yet)",
        included_ids,
        selected_chunk_ids,
    )


# ---------------------------------------------------------------------------
# External (anonymous) Q&A fast path
# ---------------------------------------------------------------------------

EXTERNAL_COMPOSER_SYSTEM_PROMPT = (
    "You are a kelp / seaweed industry analyst answering the user's "
    "question using the evidence below. The evidence is grouped per "
    "entity — each block has the entity name, a short summary, and one "
    "or more verbatim Sources passages from named reports. The Sources "
    "passages are authoritative; lean on them for specifics (numbers, "
    "dates, exact phrasing). Write in natural prose the way an analyst "
    "would — when you draw on a specific report, weave the source title "
    "into the sentence (e.g., \"according to the State of the Kelp "
    "Industry Report\").\n\n"
    "Strict grounding — answer ONLY from the evidence below. Do not add "
    "facts from prior training, even widely-known ones (company HQs, "
    "product launches, regulatory dates, geographic facts about "
    "countries / regions). The user has no way to audit prior-training "
    "claims, so they are indistinguishable from fabrication. If the "
    "evidence is insufficient, decline plainly in an analyst's voice "
    "(\"I don't have specific data on that\", \"that's outside what I "
    "track\") and stop. Do not speculate. Do not suggest the user ask "
    "again later or contact an admin.\n\n"
    "BUT — equally important: if a verbatim Source passage in the "
    "evidence directly contains what the user is asking about (a name, "
    "a fact, a relationship, a number), GIVE THE ANSWER plainly. Do "
    "NOT decline citing \"lack of specific data\" / \"I don't have that "
    "information\" when the evidence above already states the answer. "
    "Refusal is correct only when the evidence truly does not contain "
    "the answer; refusing in the presence of supporting evidence is a "
    "failure mode and worse than answering. When in doubt, answer with "
    "the citation, then add a one-line caveat about any gap.\n\n"
    "Topic-mismatch on follow-ups — judge \"enough information\" against "
    "the conversation context, not just the literal current message. "
    "Persona — you are an analyst, not a system. Don't describe your "
    "knowledge as \"my corpus\" / \"my materials\" / \"the documents I "
    "have access to\". If you must explain a limit, frame it as "
    "personal expertise coverage."
)


_REWRITE_SYSTEM_PROMPT = (
    "You are a query-rewrite helper for a retrieval-augmented chat. "
    "Given a multi-turn conversation history and a follow-up question, "
    "produce a SINGLE standalone retrieval query that captures both "
    "(a) the conversation's topic anchor and (b) what the follow-up "
    "is now asking about. The output will be embedded and matched "
    "against a corpus of industry document fragments — your job is to "
    "make sure both the topic and the new entity/intent appear in the "
    "query string.\n\n"
    "Examples:\n"
    "  HISTORY: user asks about kelp industry distribution in the "
    "Americas; assistant answers about Maine, Washington, Alaska.\n"
    "  FOLLOW-UP: \"中国呢\"\n"
    "  STANDALONE: China kelp industry distribution\n\n"
    "  HISTORY: (empty)\n"
    "  FOLLOW-UP: \"What is sugar kelp?\"\n"
    "  STANDALONE: What is sugar kelp\n\n"
    "Rules:\n"
    "- Output ONLY the standalone query as a single line. No quotes, "
    "no \"Standalone:\" preamble, no explanation, no markdown.\n"
    "- If the follow-up is already standalone, echo it verbatim.\n"
    "- Keep the query under 30 words. Mix English and the user's "
    "language as appropriate for retrieval matching."
)

_REWRITE_OUTPUT_MAX_CHARS = 400

EXTERNAL_REFUSAL_TEMPLATE = (
    "I don't have specific data on that. My expertise is the kelp / "
    "seaweed industry as represented in my available materials."
)
# The OOS post-filter previously living here was retired 2026-05-16: it
# was producing a double-message render on the streaming path (composer's
# leaked answer + horizontal rule + canned refusal) and the threat it
# defended against (LLM tool-result fabrication) was already eliminated
# by the §16.21 deterministic pipeline. Refusal correctness now relies on
# the composer prompt's grounding rules; OOS retrieval is still skipped
# in `fast_query_trace` when seeds are empty.


def _rewrite_query_with_history(client, question: str,
                                history: list[dict] | None) -> str:
    if not history:
        return question
    snapshot_lines: list[str] = []
    for m in history[-10:]:
        role = m.get("role")
        content = str(m.get("content", "")).strip()
        if role not in ("user", "assistant") or not content:
            continue
        if len(content) > 400:
            content = content[:400] + "…"
        snapshot_lines.append(f"{role.upper()}: {content}")
    if not snapshot_lines:
        return question

    user_msg = (
        "CONVERSATION HISTORY:\n"
        + "\n".join(snapshot_lines)
        + f"\n\nFOLLOW-UP: {question}\n\nSTANDALONE:"
    )
    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            thinking=False,
            timeout=30,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
    except Exception:
        return question

    if not rewritten or len(rewritten) > _REWRITE_OUTPUT_MAX_CHARS:
        return question
    rewritten = rewritten.lstrip("\"'`「『 ").rstrip("\"'`」』 ")
    for prefix in ("STANDALONE:", "Standalone:", "Query:", "查询:", "标准查询:"):
        if rewritten.upper().startswith(prefix.upper()):
            rewritten = rewritten[len(prefix):].strip()
    return rewritten or question


def _external_seeds_with_score(
    instance: GraphInstance, seed_query: str,
) -> tuple[list, float | None]:
    """Run hybrid dense+BM25 retrieval AND compute the top-1 seed score
    (kept on the return value because the trace dict still exposes it for
    eval reporting). Returns (seeds, max_seed_score)."""
    from src.graph.retrieval import _get_model, _node_text, encode_query
    seeds = top_k(
        instance.vector_store, instance.storage, seed_query, k=10,
        bm25_store=instance.bm25_store,
    )
    if not seeds:
        return [], None
    qv = encode_query(seed_query)
    model = _get_model()
    seed_vecs = model.encode(
        [_node_text(s) for s in seeds],
        convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    )
    cos = (seed_vecs @ qv).tolist()
    return seeds, max(cos) if cos else None


def fast_query_trace(instance: GraphInstance, question: str,
                     *, mode: str = "external",
                     history: list[dict] | None = None) -> dict:
    """Single-call retrieval+composer with full retrieval trace.

    Used by:
      - ``fast_query()`` — returns just the answer string
      - eval runners — need the trace dict for unified scoring

    Returned keys:
      answer (str), seed_query (str), seeds (list[{id,label,type}]),
      max_seed_score (float | None), selected_chunk_ids (list[str]),
      selected_doc_titles (list[str]), raw_answer_before_postfilter (str | None),
      llm_calls (int), timings_ms (dict).
    """
    storage = instance.storage
    client = get_client("backend")

    timings: dict[str, float] = {}
    llm_calls = 0

    t0 = time.time()
    seed_query = _rewrite_query_with_history(client, question, history)
    if seed_query != question:
        llm_calls += 1
    timings["rewrite_ms"] = int((time.time() - t0) * 1000)

    t0 = time.time()
    seeds, max_seed_score = _external_seeds_with_score(instance, seed_query)
    timings["retrieve_ms"] = int((time.time() - t0) * 1000)
    seeds_payload = [
        {"id": s.id, "label": s.label, "type": s.type} for s in seeds
    ]
    if not seeds:
        return {
            "answer": EXTERNAL_REFUSAL_TEMPLATE,
            "seed_query": seed_query,
            "seeds": [],
            "max_seed_score": max_seed_score,
            "selected_chunk_ids": [],
            "selected_doc_titles": [],
            "raw_answer_before_postfilter": None,
            "llm_calls": llm_calls,
            "timings_ms": timings,
        }

    t0 = time.time()
    ctx, included_ids, selected_chunk_ids = _build_chunk_evidence(
        instance, seeds,
        question=seed_query,
        include_neighbors=False,
        use_display_title=True,
    )
    timings["build_evidence_ms"] = int((time.time() - t0) * 1000)
    selected_doc_titles = sorted({
        display_title(getattr(getattr(s, "provenance", None), "raw_doc_id", "") or "")
        for s in seeds
    })

    touched_edge_ids = [
        e.id for e in storage.edges()
        if e.source_id in included_ids and e.target_id in included_ids
    ]
    append_query(
        storage, question=question,
        seed_node_ids=[n.id for n in seeds],
        touched_node_ids=sorted(included_ids),
        touched_edge_ids=touched_edge_ids,
    )

    # §16.1 token guard — collapse long history before composing.
    history = _maybe_summarize_history(client, history, question)

    composer_messages: list[dict] = [
        {"role": "system", "content": EXTERNAL_COMPOSER_SYSTEM_PROMPT},
    ]
    if history:
        composer_messages.extend(history)
    composer_messages.append({
        "role": "user",
        "content": f"EVIDENCE:\n{ctx}\n\nQUESTION: {question}",
    })
    t0 = time.time()
    resp = client.chat(
        messages=composer_messages, temperature=0.3, thinking=False,
    )
    timings["compose_ms"] = int((time.time() - t0) * 1000)
    llm_calls += 1
    raw_answer = resp.choices[0].message.content or EXTERNAL_REFUSAL_TEMPLATE
    return {
        "answer": raw_answer,
        "seed_query": seed_query,
        "seeds": seeds_payload,
        "max_seed_score": max_seed_score,
        "selected_chunk_ids": selected_chunk_ids,
        "selected_doc_titles": selected_doc_titles,
        # Retained for JSONL schema compatibility; the post-filter that
        # used to populate this was retired 2026-05-16.
        "raw_answer_before_postfilter": None,
        "llm_calls": llm_calls,
        "timings_ms": timings,
    }


def fast_query(instance: GraphInstance, question: str,
               *, mode: str = "external",
               history: list[dict] | None = None) -> str:
    """Single-call retrieval+composer for external users.

    Uses §16.20 hybrid dense+BM25 retrieval over node summaries.

    Thin wrapper around :func:`fast_query_trace` that drops the trace.
    """
    return fast_query_trace(
        instance, question, mode=mode, history=history,
    )["answer"]


def fast_query_stream(instance: GraphInstance, question: str,
                      *, mode: str = "external",
                      history: list[dict] | None = None) -> Iterator[str]:
    """Streaming variant — composer tokens stream straight through, no
    post-stream rewrite (the OOS post-filter was retired 2026-05-16)."""
    storage = instance.storage
    client = get_client("backend")
    seed_query = _rewrite_query_with_history(client, question, history)
    seeds, _max_seed_score = _external_seeds_with_score(instance, seed_query)
    if not seeds:
        yield EXTERNAL_REFUSAL_TEMPLATE
        return
    ctx, included_ids, _selected_chunk_ids = _build_chunk_evidence(
        instance, seeds,
        question=seed_query,
        include_neighbors=False,
        use_display_title=True,
    )

    touched_edge_ids = [
        e.id for e in storage.edges()
        if e.source_id in included_ids and e.target_id in included_ids
    ]
    append_query(
        storage, question=question,
        seed_node_ids=[n.id for n in seeds],
        touched_node_ids=sorted(included_ids),
        touched_edge_ids=touched_edge_ids,
    )

    # §16.1 token guard — same as fast_query_trace, before streaming.
    history = _maybe_summarize_history(client, history, question)

    composer_messages: list[dict] = [
        {"role": "system", "content": EXTERNAL_COMPOSER_SYSTEM_PROMPT},
    ]
    if history:
        composer_messages.extend(history)
    composer_messages.append({
        "role": "user",
        "content": f"EVIDENCE:\n{ctx}\n\nQUESTION: {question}",
    })

    for event in client.chat_stream(
        messages=composer_messages, temperature=0.3, thinking=False,
    ):
        if event.token:
            yield event.token


# ---------------------------------------------------------------------------
# Chat router: hardcoded commands + delegate to instance.qa()
# ---------------------------------------------------------------------------

_INGEST_RE = re.compile(r"^\s*/ingest(?:\s+(.+))?\s*$", re.IGNORECASE)
_SLEEP_RE = re.compile(r"^\s*/sleep(?:\s+pass)?\s*$", re.IGNORECASE)

_STATUS_INGEST = "Processing document…"
_STATUS_SLEEP = "Running maintenance cycle…"
_STATUS_QA = "Searching the knowledge graph…"


class GraphAgent:
    """Chat router. Two responsibilities:

    1. Detect ``/ingest`` and ``/sleep`` hardcoded commands and dispatch to
       the corresponding instance method directly (no LLM in the loop).
    2. For any other user input, delegate to ``instance.qa()`` which runs
       the deterministic Q&A pipeline (see ``src.modules.m2_qa``).

    There is **no LLM-driven tool selection** anymore — the old multi-step
    agent loop is gone. This eliminates the entire class of failures where
    the LLM fabricated tool fields or leaked training-data answers.

    Admin tooling (``show_provenance``, ``list_recent_*``, etc.) is no
    longer chat-accessible; callers should invoke those as Python APIs.
    """

    def __init__(self, instance: GraphInstance, expose_tools: list[str] | None = None):
        # `expose_tools` accepted for backward compatibility with tests but
        # is ignored — the chat tool set is fixed (ingest, sleep).
        self.instance = instance

    # -- Public API --

    def call(self, user_message: str,
             history: list[dict] | None = None) -> str:
        if self.instance.sleep_pass_running:
            return "Sleep pass is currently running; chat is paused until it finishes."

        msg = (user_message or "").strip()

        # /sleep — trigger maintenance cycle
        if _SLEEP_RE.match(msg):
            try:
                result = self.instance.sleep_pass()
            except Exception as exc:
                return f"Sleep pass failed: {exc}"
            return self._format_sleep_result(result)

        # /ingest — file ingestion
        m = _INGEST_RE.match(msg)
        if m:
            arg = (m.group(1) or "").strip()
            if not arg:
                return self._format_raw_files_listing()
            return self._format_ingest_result(self._ingest_file(arg))

        # Default: Q&A pipeline
        return self.instance.qa(user_message, history=history)

    def stream_call(self, user_message: str,
                    history: list[dict] | None = None) -> Iterator[str | dict]:
        """Streaming variant. Yields status dicts + content strings."""
        if self.instance.sleep_pass_running:
            yield "Sleep pass is currently running; chat is paused until it finishes."
            return

        msg = (user_message or "").strip()

        if _SLEEP_RE.match(msg):
            yield {"status": _STATUS_SLEEP}
            try:
                result = self.instance.sleep_pass()
            except Exception as exc:
                yield f"Sleep pass failed: {exc}"
                return
            yield self._format_sleep_result(result)
            return

        m = _INGEST_RE.match(msg)
        if m:
            yield {"status": _STATUS_INGEST}
            arg = (m.group(1) or "").strip()
            if not arg:
                yield self._format_raw_files_listing()
                return
            yield self._format_ingest_result(self._ingest_file(arg))
            return

        # Q&A — call qa_stream from m2_qa
        yield {"status": _STATUS_QA}
        from src.modules.m2_qa import qa_stream
        for chunk in qa_stream(self.instance, user_message, history=history):
            yield chunk

    # -- Formatters --

    def _format_sleep_result(self, result: dict) -> str:
        if not isinstance(result, dict):
            return f"Sleep pass returned: {result!r}"
        lines = ["Sleep pass complete."]
        stats = result.get("stats") or {}
        for k in ("merge_total", "edges_pruned_total", "nodes_pruned_total",
                  "new_links_total", "reinforced_total"):
            v = stats.get(k)
            if v is not None:
                pretty = k.replace("_total", "").replace("_", " ")
                lines.append(f"  - {pretty}: {v}")
        pass_id = result.get("pass_id")
        if pass_id:
            lines.append(f"  - pass_id: {pass_id}")
        return "\n".join(lines)

    def _format_ingest_result(self, result: dict) -> str:
        if not isinstance(result, dict):
            return str(result)
        if "error" in result:
            err = result["error"]
            avail = result.get("available_files")
            if avail:
                return f"{err}\n\nAvailable files:\n  - " + "\n  - ".join(avail)
            return err
        lines = [f"Ingested: {result.get('filename')}"]
        for k in ("pages_processed", "nodes_added", "edges_added",
                  "pass2_edges_added", "nodes_fused"):
            v = result.get(k)
            if v is not None:
                pretty = k.replace("_", " ")
                lines.append(f"  - {pretty}: {v}")
        if result.get("status") == "partial":
            lines.append("  - status: partial (some super-chunks failed; check log.md)")
        return "\n".join(lines)

    def _format_raw_files_listing(self) -> str:
        info = self._list_raw_files()
        if info.get("count", 0) == 0:
            return "No files in data/raw/. Place files there and try `/ingest <filename>`."
        files = info["files"]
        lines = [f"Available files in data/raw/ ({len(files)} total). "
                 "Run `/ingest <filename>` to ingest one:"]
        for f in files:
            mark = "" if f["supported"] else "  (unsupported format)"
            lines.append(f"  - {f['name']} ({f['size_bytes']:,} bytes){mark}")
        return "\n".join(lines)

    # -- Ingest implementation (mostly kept from pre-refactor) --

    _RAW_TEXT_EXTS = {".txt", ".md", ".markdown"}
    _RAW_PDF_EXTS = {".pdf"}

    def _list_raw_files(self) -> dict:
        raw_dir = (Path("data") / "raw").resolve()
        if not raw_dir.exists():
            return {"files": [], "count": 0, "note": "data/raw/ does not exist yet"}
        supported = self._RAW_TEXT_EXTS | self._RAW_PDF_EXTS
        files: list[dict] = []
        for p in sorted(raw_dir.iterdir()):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            files.append({
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "extension": ext,
                "supported": ext in supported,
            })
        return {"files": files, "count": len(files)}

    def _ingest_file(self, raw_arg: str) -> dict:
        raw_arg = (raw_arg or "").strip().strip("'\"")
        if not raw_arg:
            return {"error": "filename is required"}
        basename = Path(raw_arg).name
        raw_dir = (Path("data") / "raw").resolve()
        target = (raw_dir / basename).resolve()
        try:
            target.relative_to(raw_dir)
        except ValueError:
            return {"error": f"refused: path escapes data/raw/ ({basename!r})"}
        if not target.exists() or not target.is_file():
            target_norm = unicodedata.normalize("NFKC", basename)
            candidates = [
                p for p in raw_dir.iterdir()
                if p.is_file() and unicodedata.normalize("NFKC", p.name) == target_norm
            ]
            if len(candidates) == 1:
                target = candidates[0]
            elif len(candidates) > 1:
                return {"error": f"ambiguous filename {basename!r}; matches: {[c.name for c in candidates]}"}
            else:
                available = sorted(p.name for p in raw_dir.iterdir() if p.is_file())
                return {"error": f"no such file: data/raw/{basename}", "available_files": available}

        ext = target.suffix.lower()
        if ext in self._RAW_TEXT_EXTS:
            try:
                text = target.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                return {"error": f"could not read as utf-8: {exc}"}
            pages = [text]
        elif ext in self._RAW_PDF_EXTS:
            try:
                logging.getLogger("pdfminer").setLevel(logging.ERROR)
                import pdfplumber
                with pdfplumber.open(str(target)) as pdf:
                    pages = [pg.extract_text() or "" for pg in pdf.pages]
            except Exception as exc:
                return {"error": f"could not extract text from PDF: {exc}"}
            if not any(p.strip() for p in pages):
                return {"error": f"PDF contained no extractable text (likely scanned/image-only): {basename}"}
        else:
            return {
                "error": f"unsupported file extension {ext!r}; "
                f"supported: {sorted(self._RAW_TEXT_EXTS | self._RAW_PDF_EXTS)}"
            }

        if len(pages) <= INGEST_SUPERCHUNK_MAX_PAGES:
            return self._ingest_pages(basename, ext, pages)
        return self._ingest_pages_split(basename, ext, pages)

    @staticmethod
    def _plan_superchunks(n_pages: int, max_pages: int,
                          overlap: int) -> list[tuple[int, int]]:
        if n_pages <= 0:
            return []
        if n_pages <= max_pages:
            return [(0, n_pages)]
        step = max(1, max_pages - overlap)
        chunks: list[tuple[int, int]] = []
        start = 0
        while start < n_pages:
            end = min(start + max_pages, n_pages)
            chunks.append((start, end))
            if end >= n_pages:
                break
            start += step
        return chunks

    def _ingest_pages_split(self, basename: str, ext: str,
                            pages: list[str]) -> dict:
        ranges = self._plan_superchunks(
            len(pages),
            INGEST_SUPERCHUNK_MAX_PAGES,
            INGEST_SUPERCHUNK_OVERLAP_PAGES,
        )
        per_chunk: list[dict] = []
        totals = {
            "pages_processed": 0, "nodes_added": 0, "edges_added": 0,
            "edges_skipped": 0, "pass2_edges_added": 0,
            "pass2_edges_dropped": 0, "nodes_fused": 0,
            "edges_reclassified": 0, "edges_reclassified_kept": 0,
            "duplicate_edges_removed": 0,
        }
        ok_count = 0
        for start, end in ranges:
            sub_pages = pages[start:end]
            sub_doc_id = f"{basename}#pages_{start + 1}_{end}"
            try:
                result = self.instance.ingest(sub_pages, raw_doc_id=sub_doc_id)
                self.instance.save()
            except Exception as exc:
                per_chunk.append({
                    "raw_doc_id": sub_doc_id,
                    "page_range": [start + 1, end],
                    "status": "error",
                    "error": str(exc),
                })
                continue
            ok_count += 1
            per_chunk.append({
                "raw_doc_id": sub_doc_id,
                "page_range": [start + 1, end],
                "status": "ok",
                "run_id": result.run_id,
                "pages_processed": result.pages_processed,
                "nodes_added": result.nodes_added,
                "edges_added": result.edges_added,
            })
            for k in totals:
                totals[k] += getattr(result, k, 0)

        return {
            "status": "ok" if ok_count == len(ranges) else "partial",
            "filename": basename,
            "extension": ext,
            "split_into_superchunks": len(ranges),
            "superchunks_ok": ok_count,
            "total_pages": len(pages),
            **totals,
            "superchunks": per_chunk,
        }

    def _ingest_pages(self, basename: str, ext: str,
                      pages: list[str]) -> dict:
        try:
            result = self.instance.ingest(pages, raw_doc_id=basename)
        except Exception as exc:
            return {"error": f"ingest failed: {exc}"}
        self.instance.save()
        return {
            "status": "ok",
            "filename": basename,
            "extension": ext,
            "run_id": result.run_id,
            "pages_processed": result.pages_processed,
            "nodes_added": result.nodes_added,
            "edges_added": result.edges_added,
            "edges_skipped": result.edges_skipped,
            "pass2_edges_added": result.pass2_edges_added,
            "pass2_edges_dropped": result.pass2_edges_dropped,
            "nodes_fused": result.nodes_fused,
            "edges_reclassified": result.edges_reclassified,
            "edges_reclassified_kept": result.edges_reclassified_kept,
            "duplicate_edges_removed": result.duplicate_edges_removed,
        }
