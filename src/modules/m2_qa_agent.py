"""M2: Agent + tool box (Phase 5).

Full §6.2 tool set. Agent is a simple single-step OpenAI tool loop —
LangGraph is reserved for Phase 6 sleep-pass orchestration (§5.0). Tools
that depend on later phases return stub payloads so the routing test can
still see them in the schema list.
"""

import json
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

from src.graph.instance import GraphInstance
from src.graph.retrieval import display_title, top_k, with_neighbors
from src.graph.tokens import (
    INGEST_INPUT_MAX_TOKENS,
    INGEST_SUPERCHUNK_MAX_PAGES,
    INGEST_SUPERCHUNK_OVERLAP_PAGES,
    count_tokens,
)
from src.graph.traversal_log import append_query
from src.llm.local_client import StreamEvent
from src.llm.routing import get_client

SYSTEM_PROMPT = (
    "You are a kelp / seaweed industry analyst. Your working memory holds an "
    "evolving corpus of industry reports, scientific studies, and impact "
    "assessments on the sector. When the user asks about the industry — "
    "trends, economics, players, ecology, technology — answer as a domain "
    "expert synthesizing that evidence. Speak in natural prose, not as a "
    "system reading off a database; when you lean on a specific report, "
    "weave the source into the sentence (\"according to the State of the "
    "Kelp Industry Report...\") rather than citing internal IDs.\n\n"
    "You have a tool box for retrieving evidence, running maintenance on "
    "your working memory, and surfacing exact provenance when asked. Use "
    "whichever tool fits the request. If the user is just chatting and no "
    "tool is needed, reply in natural language.\n\n"
    "Five hard rules — non-negotiable regardless of phrasing:\n"
    "1. New evidence only enters your working memory when the user places a "
    "document under data/raw/ and asks you to ingest it — call `ingest_file` "
    "then. You CANNOT write facts directly from chat; do not pretend otherwise.\n"
    "2. If the user asks what's available to ingest, call `list_raw_files` — "
    "do not claim you cannot see the directory.\n"
    "3. If the user pushes for exact provenance (\"where did you read this?\", "
    "\"show me the source\"), call `show_provenance` and quote the raw "
    "document rather than paraphrasing.\n"
    "4. Be faithful to tool returns. When a tool's response does not contain "
    "the field the user is asking about, say so plainly — \"the log does not "
    "record that field\" or \"that tool doesn't expose this\" — and suggest a "
    "question you can answer. NEVER invent labels, ids, weights, or other "
    "values to fill a gap, even if the guess sounds reasonable.\n"
    "5. Working memory is your ONLY source of industry knowledge. ANY "
    "question about the kelp / seaweed industry — companies, products, "
    "trends, economics, technology, geographies, people, regulations, "
    "competitive landscape — MUST be answered from working memory only. "
    "Mandatory flow: call `graph_query` first, then answer from that "
    "evidence. If the evidence doesn't cover the specific question, say "
    "plainly that working memory does not contain that information and "
    "offer to ingest a relevant report. Do NOT fill gaps with information "
    "from your prior training — the user cannot audit prior-training facts "
    "via `show_provenance`, so they are indistinguishable from fabrication "
    "and undermine the whole point of having a corpus. If you find yourself "
    "\"knowing\" a fact (e.g. that company X is based in country Y) without "
    "tracing it to a working-memory node, treat it as suspect: either "
    "re-query with `graph_query` to confirm, or say you don't have that "
    "information. Mixing prior-training claims with corpus claims in the "
    "same answer — even when both happen to be true — is a hard violation, "
    "because the user is reading them as one stream of grounded analysis."
)

# Budget for graph_query retrieval context (§12.5.2). With 128K context
# (§ changelog 2026-05-05) the cap is generous; the limit exists to guard
# against pathological cases (a hub entity with 100 source chunks would
# otherwise dominate the prompt).
RETRIEVAL_BUDGET_TOKENS = 20_000


_RERANK_TOP_N_CHUNKS = 10  # task 1: top-N chunks after rerank


def _build_chunk_evidence(
    instance: "GraphInstance",
    seeds: list,
    *,
    question: str,
    include_neighbors: bool,
    use_display_title: bool,
    top_n_chunks: int = _RERANK_TOP_N_CHUNKS,
    budget_tokens: int = RETRIEVAL_BUDGET_TOKENS,
) -> tuple[str, set[str]]:
    """Build the EVIDENCE section for a composer prompt (§16.17.6 + task 1 rerank).

    Two-stage retrieval:
      1. ``seeds`` is the entity-level recall (top-K from FAISS over node
         summaries, K=10 internal/external as of task 1).
      2. Every chunk attached to those entities (via ``text_unit_ids``) is
         re-scored against ``question`` by cosine on the same
         sentence-transformers embedding used for entity recall. The top
         ``top_n_chunks`` (subject to ``budget_tokens``) make it into the
         EVIDENCE block, ranked by relevance instead of extraction order.

    Each seed renders as:
        Entity: <label> (<type>)
        Summary: <summary>
        Sources:
          [from "<doc title>", page N]
          <verbatim chunk text>
          ...

    Within a seed, surviving chunks are ordered by their global rerank
    score (most relevant first). Seeds whose chunks all lose the rerank
    still render with label + summary — useful as an indication that the
    entity matched but no chunk was strongly relevant.

    Trailing NEIGHBORS section (internal only) lists 1-hop neighbors with
    summary only — no chunk injection — to give the agent loop pivot
    points without exploding the prompt.

    Returns (evidence_text, included_node_ids). included_node_ids feeds
    the traversal-log reinforcement and is the set of nodes that actually
    made it into the prompt (not just into the seeds list).
    """
    storage = instance.storage
    text_units = instance.text_units

    # ---- Stage 1: collect candidate chunks across seeds (deduped). ----
    # We need per-seed groupings AND a flat list for rerank.
    seed_chunk_ids: dict[str, list[str]] = {}     # seed.id -> ordered chunk_ids that exist
    chunk_payloads: dict[str, dict] = {}          # chunk_id -> chunk dict
    for seed in seeds:
        ids: list[str] = []
        for cid in (getattr(seed, "text_unit_ids", None) or []):
            cd = text_units.get(cid)
            if cd is None:
                continue
            ids.append(cid)
            chunk_payloads.setdefault(cid, cd)
        seed_chunk_ids[seed.id] = ids

    # ---- Stage 2: rerank chunks against the question by cosine. ----
    # Embed once with the same sentence-transformers model that built FAISS
    # (paraphrase-multilingual-MiniLM-L12-v2 — cross-lingual capable).
    # On-the-fly chunk embedding cost ~5-20ms per chunk; for typical pools
    # of 30-150 chunks this adds ~0.5-3s before the composer LLM call.
    chunk_rank: dict[str, int] = {}   # chunk_id -> rerank position (0 = best)
    all_chunk_ids = list(chunk_payloads.keys())
    if all_chunk_ids:
        from src.graph.retrieval import _get_model, encode_query
        qv = encode_query(question)
        model = _get_model()
        chunk_texts = [
            (chunk_payloads[cid].get("text", "") or "")
            for cid in all_chunk_ids
        ]
        chunk_vecs = model.encode(
            chunk_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        # Cosine = dot (vectors normalized). Higher = more relevant.
        scores = (chunk_vecs @ qv).tolist()
        ranked = sorted(
            zip(all_chunk_ids, scores),
            key=lambda x: -x[1],
        )
        chunk_rank = {cid: i for i, (cid, _) in enumerate(ranked)}

    # ---- Stage 3: select top-N chunks within token budget. ----
    # Walk the rerank order, accept each chunk if it fits the remaining
    # budget and we haven't hit top_n_chunks yet. Skip-don't-break on
    # oversized chunks — a single 8K-token chunk shouldn't lock the rest
    # out, the next candidate might fit. Cap by N OR by tokens, whichever
    # hits first.
    selected_chunk_ids: set[str] = set()
    chunk_render_cache: dict[str, tuple[str, int]] = {}  # cid -> (rendered_text, token_count)
    used_tokens = 0
    for cid, _score in sorted(chunk_rank.items(), key=lambda x: x[1]):
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
            continue  # try next — a smaller one might still fit
        chunk_render_cache[cid] = (rendered, ct)
        selected_chunk_ids.add(cid)
        used_tokens += ct

    # ---- Stage 4: render entity-grouped EVIDENCE. ----
    blocks: list[str] = []
    included_ids: set[str] = set()
    for seed in seeds:
        block_lines = [
            f"Entity: {seed.label} ({seed.type})",
            f"Summary: {seed.summary}",
        ]
        surviving_for_seed = [
            cid for cid in seed_chunk_ids.get(seed.id, [])
            if cid in selected_chunk_ids
        ]
        # Order surviving chunks by rerank position so the most relevant
        # passage appears first inside the seed block.
        surviving_for_seed.sort(key=lambda cid: chunk_rank.get(cid, 1 << 30))
        if surviving_for_seed:
            block_lines.append("Sources:")
            for cid in surviving_for_seed:
                rendered, _ = chunk_render_cache[cid]
                block_lines.append(rendered)
        block_text = "\n".join(block_lines)
        bt = count_tokens(block_text)
        # Budget guard for the entity headers themselves (summary lines
        # could push us over on a very tight budget). Stop adding seed
        # blocks once headers alone would overflow.
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
            blocks.append("\nNEIGHBORS (context only, no full sources):\n" + "\n".join(nbr_lines))

    text = "\n\n".join(blocks).strip()
    return text or "(no evidence in working memory yet)", included_ids

# Max tool-call iterations inside a single `agent.call()` (one user message).
# Distinct from sleep-pass's MERGE_MAX_ITER — that's M4 convergence, this is
# M2 chat budget. Sized to comfortably handle batch ingest workflows
# (e.g., list_raw_files + N ingest_file + a wrap-up). Hitting this cap means
# either a real long batch or the model is stuck calling the same tool —
# the runaway guard below catches the latter case earlier.
AGENT_MAX_TOOL_STEPS = 30
# If the model issues this many consecutive identical (name, args) calls,
# break the loop — it's stuck and won't make progress.
AGENT_DUPLICATE_CALL_LIMIT = 3

# Path of the shared sleep-pass log (see m4_sleep_pass.pass_log).
_PASS_LOG_PATH = Path("log.md")


def _tail_log_events(kind: str, k: int) -> list[dict]:
    """Return the last k JSON payloads in log.md whose kind matches.

    log.md lines are of the form: '- [ts] kind summary | {json}'. We just
    parse the JSON trailer.
    """
    import json as _json
    if not _PASS_LOG_PATH.exists():
        return []
    out: list[dict] = []
    for line in _PASS_LOG_PATH.read_text(encoding="utf-8").splitlines()[::-1]:
        idx = line.rfind("| ")
        if idx < 0:
            continue
        try:
            payload = _json.loads(line[idx + 2:])
        except _json.JSONDecodeError:
            continue
        if payload.get("kind") == kind:
            out.append(payload)
            if len(out) >= k:
                break
    return out


# --- Tool schemas ------------------------------------------------------

def _tool_schemas() -> dict[str, dict]:
    return {
        "graph_query": {
            "type": "function",
            "function": {
                "name": "graph_query",
                "description": (
                    "Query the knowledge graph with a natural language "
                    "question. Use this when the user asks factual questions "
                    "about entities or relationships in the graph."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The user's question in natural language"},
                    },
                    "required": ["question"],
                },
            },
        },
        "trigger_sleep_pass": {
            "type": "function",
            "function": {
                "name": "trigger_sleep_pass",
                "description": (
                    "Run a sleep pass (prune, consolidate, reinforce, link "
                    "formation) over the graph. Use when the user explicitly "
                    "asks to run maintenance, consolidation, or a sleep pass."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        "get_graph_stats": {
            "type": "function",
            "function": {
                "name": "get_graph_stats",
                "description": "Return node count, edge count, and type distribution. Use when the user asks about graph size or composition.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        "list_recent_merges": {
            "type": "function",
            "function": {
                "name": "list_recent_merges",
                "description": "List the most recent entity merges from sleep passes. Use when the user asks what was merged recently.",
                "parameters": {
                    "type": "object",
                    "properties": {"k": {"type": "integer", "description": "How many recent merges to return", "default": 10}},
                    "required": [],
                },
            },
        },
        "list_recent_prunings": {
            "type": "function",
            "function": {
                "name": "list_recent_prunings",
                "description": "List the most recent edge/node prunings from sleep passes. Use when the user asks about what was deleted or pruned recently.",
                "parameters": {
                    "type": "object",
                    "properties": {"k": {"type": "integer", "description": "How many recent prunings to return", "default": 10}},
                    "required": [],
                },
            },
        },
        "show_provenance": {
            "type": "function",
            "function": {
                "name": "show_provenance",
                "description": (
                    "Show the provenance and source-text excerpts for a "
                    "specific node OR edge given its UUID. The result "
                    "includes raw_doc_id, page numbers, and the verbatim "
                    "page text the entity / relationship was extracted "
                    "from — quote it directly to the user when they ask "
                    "for a citation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"node_or_edge_id": {"type": "string", "description": "The UUID of the node or edge"}},
                    "required": ["node_or_edge_id"],
                },
            },
        },
        "read_ontology": {
            "type": "function",
            "function": {
                "name": "read_ontology",
                "description": "Read the ontology schema document. Use when the user asks what entity or relation types exist, or about merge / extraction rules.",
                "parameters": {
                    "type": "object",
                    "properties": {"section": {"type": "string", "description": "Optional section heading to read only that section"}},
                    "required": [],
                },
            },
        },
        "run_plant_recover_eval": {
            "type": "function",
            "function": {
                "name": "run_plant_recover_eval",
                "description": "Run the plant-and-recover quantitative evaluation on the experiment instance. Use when the user asks to evaluate derived-edge recall on planted relations.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        "compare_with_baseline": {
            "type": "function",
            "function": {
                "name": "compare_with_baseline",
                "description": "Run the same evaluation against the baseline RAG system and return a comparison report.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        "list_raw_files": {
            "type": "function",
            "function": {
                "name": "list_raw_files",
                "description": (
                    "List the files currently sitting in data/raw/ that are "
                    "available to ingest. Use this whenever the user asks "
                    "what files are available, what's in data/raw/, or which "
                    "documents you can ingest. Returns basenames, sizes, and "
                    "extensions; supported formats for ingest are .txt, .md, "
                    "and .pdf."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        "ingest_file": {
            "type": "function",
            "function": {
                "name": "ingest_file",
                "description": (
                    "Load an authoritative raw document from data/raw/ and "
                    "ingest it via M1 (DirectProv). This is the ONLY way "
                    "knowledge enters the graph — call it whenever the user "
                    "asks to load / import / ingest a file or document by name. "
                    "Supported formats: .txt, .md (read as UTF-8) and .pdf "
                    "(text extracted via pdfplumber). PDFs are processed page "
                    "by page with cross-page entity carry-over and a "
                    "second-pass relationship sweep. Documents longer than "
                    "80 pages are auto-split into super-chunks (1-page "
                    "overlap) and each super-chunk runs the full M1 pipeline; "
                    "the result reports totals across the entire document."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "File basename under data/raw/ (e.g., 'q1_report.pdf'). Path components are stripped for safety.",
                        },
                    },
                    "required": ["filename"],
                },
            },
        },
    }


ALL_TOOL_NAMES = list(_tool_schemas().keys())
DEFAULT_TOOLS = list(ALL_TOOL_NAMES)


# --- Text-format tool-call fallback ------------------------------------
#
# Some models (e.g. Gemma) emit tool calls as text — `<|tool_call|>...`,
# `<tool_call>...</tool_call>`, raw JSON, or `call:name{args}` — instead
# of populating the OpenAI `tool_calls` field. This parser covers the
# common shapes so the agent keeps working regardless of backend (LM
# Studio, llama.cpp, vLLM) or chat-template version.

_TOOL_CALL_BLOCK_RE = re.compile(
    r"<\|?\s*tool_call\s*\|?>(.+?)<\|?\s*/?\s*tool_call\s*\|?>",
    re.DOTALL,
)
_GEMMA_QUOTE_RE = re.compile(r"<\|\"\|>")
_CALL_BODY_RE = re.compile(
    r"(?:call\s*:\s*)?([A-Za-z_]\w*)\s*\{(.*)\}",
    re.DOTALL,
)
_BARE_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_]\w*)\s*:")


def _parse_text_tool_call(content: str, allowed: list[str]) -> tuple[str, dict] | None:
    if not content:
        return None
    m = _TOOL_CALL_BLOCK_RE.search(content)
    body = m.group(1).strip() if m else content.strip()

    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        name = obj.get("name") or obj.get("function")
        args = obj.get("arguments") or obj.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if isinstance(name, str) and name in allowed and isinstance(args, dict):
            return name, args

    m = _CALL_BODY_RE.search(body)
    if not m:
        return None
    name = m.group(1)
    if name not in allowed:
        return None
    raw = "{" + m.group(2) + "}"
    raw = _GEMMA_QUOTE_RE.sub('"', raw)
    raw = _BARE_KEY_RE.sub(r'\1"\2":', raw)
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        return name, {}
    return (name, args) if isinstance(args, dict) else None


def _synthesize_tool_call(name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id="fb_" + uuid.uuid4().hex[:8],
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _scrub_tool_call_markers(content: str) -> str:
    """Remove any leftover `<|tool_call|>...<|tool_call|>` blocks from text
    that's about to be shown to the user. Defense-in-depth — even if the
    fallback parser missed a variant, the user shouldn't see raw markup.
    Returns "(empty reply)" if scrubbing leaves nothing meaningful."""
    if not content:
        return content
    cleaned = _TOOL_CALL_BLOCK_RE.sub("", content).strip()
    return cleaned if cleaned else "(empty reply — model emitted only tool-call markup; please retry)"


# --- Agent -------------------------------------------------------------

@dataclass
class RouteResult:
    tool_name: str | None
    tool_args: dict | None
    text: str | None
    raw_args: str | None
    error: str | None = None


class GraphAgent:
    def __init__(
        self,
        instance: GraphInstance,
        expose_tools: list[str] | None = None,
    ):
        self.instance = instance
        self._all_schemas = _tool_schemas()
        names = expose_tools if expose_tools is not None else DEFAULT_TOOLS
        self.schemas = [self._all_schemas[n] for n in names if n in self._all_schemas]
        self.exposed_names = [n for n in names if n in self._all_schemas]
        self.client = get_client("agent")
        self._impls: dict[str, Callable[[dict], Any]] = self._build_impls()

    # -- Tool implementations --

    def _build_impls(self) -> dict[str, Callable[[dict], Any]]:
        gi = self.instance
        return {
            "graph_query": self._impl_graph_query,
            "trigger_sleep_pass": lambda a: gi.sleep_pass(),
            "get_graph_stats": lambda a: gi.storage.stats(),
            "list_recent_merges": self._impl_list_recent_merges,
            "list_recent_prunings": self._impl_list_recent_prunings,
            "show_provenance": self._impl_show_provenance,
            "read_ontology": self._impl_read_ontology,
            "run_plant_recover_eval": lambda a: {"status": "not_implemented"},
            "compare_with_baseline": lambda a: {"status": "not_implemented"},
            "list_raw_files": self._impl_list_raw_files,
            "ingest_file": self._impl_ingest_file,
        }

    def _impl_list_recent_merges(self, args: dict) -> dict:
        k = int(args.get("k", 10))
        return {"merges": _tail_log_events("merge", k)}

    def _impl_list_recent_prunings(self, args: dict) -> dict:
        k = int(args.get("k", 10))
        return {"prunings": _tail_log_events("prune", k)}

    def _impl_graph_query(self, args: dict) -> dict:
        q = str(args.get("question", ""))
        storage = self.instance.storage
        # Safety net: even when the agent's outer LLM has the full history
        # in its messages, the question it passes via tool args may still
        # be deictic ("中国呢"). Use a small dedicated rewrite call
        # (thinking=False, ~1-2s) to produce a standalone retrieval query
        # that includes both the conversation's topic anchor and the new
        # entity. Same mechanism as fast_query — internal and external
        # behave consistently on follow-ups.
        history = getattr(self, "_turn_history", None) or []
        seed_q = _rewrite_query_with_history(self.client, q, history)
        # Task 1 (2026-05-11): widen entity recall to k=10 to feed the
        # chunk reranker a larger pool. Chunks themselves are reranked by
        # cosine against `seed_q` inside `_build_chunk_evidence`, then
        # top-N (default 10) within the token budget make it to prompt.
        # Neighbor expansion still runs (summary-only context for the
        # agent loop's pivot decisions).
        seeds = top_k(self.instance.vector_store, storage, seed_q, k=10)
        ctx, included_ids = _build_chunk_evidence(
            self.instance, seeds,
            question=seed_q,
            include_neighbors=True,
            use_display_title=False,  # internal cites raw_doc_id directly
        )
        # Log everything that made it into context — 4c uses this to reinforce.
        touched_edge_ids = [
            e.id for e in storage.edges()
            if e.source_id in included_ids and e.target_id in included_ids
        ]
        append_query(
            storage,
            question=q,
            seed_node_ids=[n.id for n in seeds],
            touched_node_ids=sorted(included_ids),
            touched_edge_ids=touched_edge_ids,
        )
        resp = self.client.chat(
            messages=[
                {"role": "system", "content": (
                    "You are a kelp / seaweed industry analyst answering the "
                    "user's question using the evidence below. The evidence "
                    "is grouped per entity: each block has the entity name "
                    "and type, a brief summary, and one or more verbatim "
                    "Sources passages from named documents. Quote and cite "
                    "from the verbatim passages whenever specific facts "
                    "(numbers, dates, exact phrasing) are relevant — they "
                    "are the authoritative version. Write in natural prose "
                    "the way an industry analyst would; when you draw on a "
                    "specific report, weave the source title into the "
                    "sentence (e.g., \"according to the State of the Kelp "
                    "Industry Report...\"). Drop any .pdf suffix when "
                    "naming a source.\n\n"
                    "After the per-entity blocks you may see a NEIGHBORS "
                    "section listing 1-hop graph neighbors with summaries "
                    "only (no source passages). Use neighbors as context "
                    "to spot related entities you might want to reason "
                    "about — but do NOT cite specific facts from a "
                    "neighbor's summary as if it were primary evidence.\n\n"
                    "Strict grounding rule: answer ONLY from the evidence "
                    "below. Do NOT add facts you happen to know from prior "
                    "training — even widely-known facts about companies, "
                    "geographies, products, regulations. The caller will "
                    "audit this answer against the source documents; "
                    "anything you add beyond the evidence is fabrication "
                    "to them. If the evidence is insufficient to answer "
                    "the specific question asked, say so plainly and stop "
                    "— do NOT pad with general knowledge to look helpful. "
                    "Topic-mismatch case: when the caller's QUESTION is a "
                    "follow-up that inherits a topic from prior turns "
                    "(e.g., the conversation is about the kelp industry "
                    "and the question is \"what about China?\"), evidence "
                    "that mentions the new entity (China) in an "
                    "off-topic context (e.g., general aquaculture or "
                    "capture fisheries) is NOT sufficient — say plainly "
                    "that the evidence doesn't cover the kelp industry "
                    "for that entity. Do not silently pivot to the "
                    "off-topic data."
                )},
                {"role": "user", "content": f"EVIDENCE:\n{ctx}\n\nQUESTION: {q}"},
            ],
            temperature=0.2,
        )
        return {
            "answer": resp.choices[0].message.content or "",
            "citations": [{"id": n.id, "label": n.label} for n in seeds],
        }

    def _impl_show_provenance(self, args: dict) -> dict:
        """Return provenance for a node OR an edge (§16.8.4 + §16.17.6).

        For both kinds, resolve any text_unit_ids back to the verbatim
        chunk text so the agent can quote it directly to the user without
        needing a separate fetch tool. For edges this also surfaces the
        evidence_quote if PASS 2 emitted one.
        """
        tgt = str(args.get("node_or_edge_id", ""))
        n = self.instance.storage.get_node(tgt)
        if n is not None:
            payload = {
                "kind": "node",
                "id": n.id,
                "label": n.label,
                "type": n.type,
                "provenance": n.provenance.model_dump(),
            }
            sources = self._resolve_text_units(n.text_unit_ids)
            if sources:
                payload["sources"] = sources
            return payload

        e = self.instance.storage.get_edge(tgt)
        if e is not None:
            src_node = self.instance.storage.get_node(e.source_id)
            tgt_node = self.instance.storage.get_node(e.target_id)
            payload = {
                "kind": "edge",
                "id": e.id,
                "type": e.type,
                "source": {
                    "id": e.source_id,
                    "label": src_node.label if src_node else None,
                },
                "target": {
                    "id": e.target_id,
                    "label": tgt_node.label if tgt_node else None,
                },
                "provenance": e.provenance.model_dump(),
            }
            if e.evidence_quote:
                payload["evidence_quote"] = e.evidence_quote
            sources = self._resolve_text_units(e.text_unit_ids)
            if sources:
                payload["sources"] = sources
            return payload

        return {"kind": "not_found", "id": tgt}

    def _resolve_text_units(self, chunk_ids: list[str]) -> list[dict]:
        """Map chunk_ids → list of {raw_doc_id, page_num, text} dicts. Used
        by show_provenance to inline the source text into its result."""
        if not chunk_ids:
            return []
        out: list[dict] = []
        for cd in self.instance.text_units.get_many(chunk_ids):
            out.append({
                "raw_doc_id": cd.get("raw_doc_id", ""),
                "page_num": cd.get("page_num"),
                "text": cd.get("text", ""),
            })
        return out

    def _impl_read_ontology(self, args: dict) -> dict:
        p = Path(self.instance.ontology_path)
        if not p.exists():
            return {"content": "", "note": "ontology.md not present yet"}
        text = p.read_text(encoding="utf-8")
        section = args.get("section")
        if section:
            for block in text.split("\n## "):
                if block.lower().startswith(str(section).lower()):
                    return {"content": "## " + block}
        return {"content": text}

    _RAW_TEXT_EXTS = {".txt", ".md", ".markdown"}
    _RAW_PDF_EXTS = {".pdf"}

    def _impl_list_raw_files(self, args: dict) -> dict:
        """List files sitting in data/raw/ so the agent can pick targets."""
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

    def _impl_ingest_file(self, args: dict) -> dict:
        """Read data/raw/<basename> and ingest it via M1 (§16.17).

        Path traversal is prevented by reducing to ``basename`` and then
        asserting the resolved path sits inside the data/raw/ directory.
        PDFs are split per page; non-paginated text formats become a
        single-element pages list. M1's ingest_document handles the
        two-pass extraction internally — no chunk planning happens here.
        """
        raw_arg = str(args.get("filename", "")).strip()
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
            # Fuzzy fallback: NFKC-normalize both sides and look for a unique
            # match. The LLM tends to emit half-width `()` in tool args even
            # when the on-disk filename uses full-width `（）` — without this
            # fallback the user has to retype the filename verbatim every
            # time. Same trick covers other CJK width / compatibility chars.
            target_norm = unicodedata.normalize("NFKC", basename)
            candidates = [
                p for p in raw_dir.iterdir()
                if p.is_file()
                and unicodedata.normalize("NFKC", p.name) == target_norm
            ]
            if len(candidates) == 1:
                target = candidates[0]
            elif len(candidates) > 1:
                return {
                    "error": (
                        f"ambiguous filename {basename!r}; multiple files "
                        f"normalize to the same name: {[c.name for c in candidates]}"
                    )
                }
            else:
                # Hand back the actual directory contents so the agent can
                # pick a real name on the next turn rather than guessing
                # again. Without this, "no such file: X" is a dead end —
                # the model has no signal to recover from a wrong filename.
                available = sorted(
                    p.name for p in raw_dir.iterdir() if p.is_file()
                )
                return {
                    "error": f"no such file: data/raw/{basename}",
                    "available_files": available,
                    "hint": (
                        "filename must match exactly one entry in "
                        "available_files; call list_raw_files first if "
                        "you're unsure"
                    ),
                }

        ext = target.suffix.lower()

        if ext in self._RAW_TEXT_EXTS:
            try:
                text = target.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                return {"error": f"could not read as utf-8: {exc}"}
            pages = [text]
        elif ext in self._RAW_PDF_EXTS:
            try:
                import logging
                # pdfminer.six (pdfplumber's backend) spams "Could not get
                # FontBBox" warnings on many real-world PDFs — purely
                # cosmetic, suppress so chat logs stay readable.
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
                "error": (
                    f"unsupported file extension {ext!r}; "
                    f"supported: {sorted(self._RAW_TEXT_EXTS | self._RAW_PDF_EXTS)}"
                )
            }

        # §16.17 super-chunk wrapper (2026-05-10): a 200-page PDF would
        # blow up M1's `prior_entities_block` past the 128K context
        # window in late PASS 1 / all of PASS 2. Slice anything bigger
        # than INGEST_SUPERCHUNK_MAX_PAGES into bounded super-chunks
        # with a 1-page overlap; each runs the full M1 pipeline as if
        # it were its own document. Cross-super-chunk same-entity
        # duplicates are accepted and resolved by M4b sleep pass.
        if len(pages) <= INGEST_SUPERCHUNK_MAX_PAGES:
            return self._ingest_pages(basename, ext, pages)
        return self._ingest_pages_split(basename, ext, pages)

    @staticmethod
    def _plan_superchunks(
        n_pages: int, max_pages: int, overlap: int,
    ) -> list[tuple[int, int]]:
        """Produce half-open page ranges (start, end) covering ``n_pages``.

        Each range has at most ``max_pages`` pages; consecutive ranges
        share ``overlap`` pages on the boundary. Empty input → empty list.
        For ``n_pages <= max_pages`` returns a single range covering all.

        Examples (max=80, overlap=1):
            n=80   → [(0, 80)]
            n=81   → [(0, 80), (79, 81)]
            n=200  → [(0, 80), (79, 159), (158, 200)]
        """
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

    def _ingest_pages_split(
        self, basename: str, ext: str, pages: list[str],
    ) -> dict:
        """Run M1 over a long PDF in bounded super-chunks. Each super-chunk
        becomes its own ``raw_doc_id`` (``basename#pages_X_Y``) so chunks
        and provenance can be attributed back to the source slice. Stats
        are summed across super-chunks; per-super-chunk detail is also
        returned for audit. Failures of one super-chunk are logged and
        skipped — the rest still run."""
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
                "pass2_edges_added": result.pass2_edges_added,
                "edges_reclassified": result.edges_reclassified,
                "duplicate_edges_removed": result.duplicate_edges_removed,
                "nodes_fused": result.nodes_fused,
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
            "page_overlap": INGEST_SUPERCHUNK_OVERLAP_PAGES,
            "superchunk_max_pages": INGEST_SUPERCHUNK_MAX_PAGES,
            **totals,
            "superchunks": per_chunk,
        }

    def _ingest_pages(self, basename: str, ext: str, pages: list[str]) -> dict:
        """Run M1 on a pages list (§16.17). Single result shape regardless
        of whether the source is multi-page PDF or single-blob text."""
        try:
            result = self.instance.ingest(pages, raw_doc_id=basename)
        except Exception as exc:
            return {"error": f"ingest failed: {exc}"}
        self.instance.save()
        out = {
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
        if result.page_warnings:
            out["page_warnings"] = result.page_warnings
        return out

    # -- Public API --

    def route(self, user_message: str, history: list[dict] | None = None) -> RouteResult:
        """Single LLM call; return tool selection without executing."""
        if self.instance.sleep_pass_running:
            return RouteResult(None, None, "Sleep pass is currently running; chat is paused until it finishes.", None)
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        try:
            resp = self.client.chat(
                messages=messages,
                tools=self.schemas,
                tool_choice="auto",
                temperature=0.2,
                timeout=120,
            )
        except Exception as exc:
            return RouteResult(None, None, None, None, error=str(exc))
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            fb = _parse_text_tool_call(msg.content or "", self.exposed_names)
            if fb is not None:
                calls = [_synthesize_tool_call(*fb)]
        if calls:
            first = calls[0]
            raw = first.function.arguments or "{}"
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            return RouteResult(first.function.name, parsed, msg.content, raw)
        return RouteResult(None, None, msg.content, None)

    def call(self, user_message: str, history: list[dict] | None = None) -> str:
        """Multi-step tool loop: route → execute → feed result back → repeat
        until the model returns a tool-free natural-language reply or
        MAX_STEPS is reached. Works with both structured `tool_calls` and
        text-format fallbacks (see `_parse_text_tool_call`)."""
        if self.instance.sleep_pass_running:
            return "Sleep pass is currently running; chat is paused until it finishes."
        # Stash for tool impls to enrich retrieval seeds with conversation
        # context (see _impl_graph_query). Cleared after the call returns.
        self._turn_history = history or []
        self._turn_user_message = user_message
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        last_signature: tuple[str, str] | None = None
        duplicate_streak = 0
        for step in range(AGENT_MAX_TOOL_STEPS):
            resp = self.client.chat(
                messages=messages,
                tools=self.schemas,
                tool_choice="auto",
                temperature=0.2,
                timeout=120,
            )
            msg = resp.choices[0].message
            calls = msg.tool_calls or []
            if not calls:
                fb = _parse_text_tool_call(msg.content or "", self.exposed_names)
                if fb is None:
                    # No structured call, no parseable text call — model is
                    # done. Scrub any leftover marker-shaped text so the user
                    # never sees raw `<|tool_call|>...` blocks (defense in
                    # depth — the parser SHOULD have caught real calls above).
                    return _scrub_tool_call_markers(msg.content or "")
                calls = [_synthesize_tool_call(*fb)]

            first = calls[0]
            name = first.function.name
            raw_args = first.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            # Runaway guard: same tool + same args N times in a row → break.
            signature = (name, raw_args)
            if signature == last_signature:
                duplicate_streak += 1
                if duplicate_streak >= AGENT_DUPLICATE_CALL_LIMIT:
                    return (
                        f"Stopped: model called `{name}` with the same "
                        f"arguments {duplicate_streak + 1} times in a row "
                        f"and isn't making progress. Last result is in the "
                        f"trace; please rephrase or split your request."
                    )
            else:
                duplicate_streak = 0
            last_signature = signature

            impl = self._impls.get(name)
            result = impl(args) if impl else {"error": f"unknown tool {name}"}

            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": first.id,
                            "type": "function",
                            "function": {"name": name, "arguments": raw_args},
                        }
                    ],
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": first.id, "content": json.dumps(result, default=str)}
            )

        # Step budget exhausted — force a tool-free summary so the user
        # gets *something* coherent instead of another dangling tool call.
        # The bare-LLM call has no `tools=` arg, so Gemma sometimes still
        # emits a tool_call shape in text. Try the parser one last time —
        # if it lands, execute it and let *that* result become the reply;
        # otherwise scrub markers and return whatever prose came back.
        final = self.client.chat(messages=messages, temperature=0.2, timeout=120)
        final_content = final.choices[0].message.content or ""
        fb = _parse_text_tool_call(final_content, self.exposed_names)
        if fb is not None:
            name, args = fb
            impl = self._impls.get(name)
            if impl is not None:
                result = impl(args)
                return f"[{name}] {json.dumps(result, default=str, ensure_ascii=False)}"
        return _scrub_tool_call_markers(final_content)

    # -- streaming variant (§16.15.2) ------------------------------------

    _TOOL_STATUS_MAP: dict[str, str] = {
        "graph_query": "Searching the knowledge graph…",
        "get_graph_stats": "Checking graph statistics…",
        "show_provenance": "Looking up source information…",
        "list_recent_merges": "Reviewing recent changes…",
        "list_recent_prunings": "Reviewing recent changes…",
        "list_raw_files": "Checking available documents…",
        "ingest_file": "Processing document…",
        "trigger_sleep_pass": "Running maintenance cycle…",
        "read_ontology": "Reading domain schema…",
        "run_plant_recover_eval": "Running evaluation…",
        "compare_with_baseline": "Running baseline comparison…",
    }

    def stream_call(
        self,
        user_message: str,
        history: list[dict] | None = None,
    ) -> Iterator[str | dict]:
        """Streaming variant of :meth:`call` (§16.15.2).

        Yields a mix of:
          - ``{"status": "…"}`` — status updates for the UI.
          - ``str`` — content tokens of the final natural-language reply.

        Tool-loop intermediate steps use synchronous ``client.chat()``.
        The final reply is streamed token-by-token via
        ``client.chat_stream()``.
        """
        if self.instance.sleep_pass_running:
            yield "Sleep pass is currently running; chat is paused until it finishes."
            return

        self._turn_history = history or []
        self._turn_user_message = user_message
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        last_signature: tuple[str, str] | None = None
        duplicate_streak = 0

        for step in range(AGENT_MAX_TOOL_STEPS):
            yield {"status": "Thinking…"}

            resp = self.client.chat(
                messages=messages,
                tools=self.schemas,
                tool_choice="auto",
                temperature=0.2,
                timeout=120,
            )
            msg = resp.choices[0].message
            calls = msg.tool_calls or []
            if not calls:
                fb = _parse_text_tool_call(msg.content or "", self.exposed_names)
                if fb is None:
                    break
                calls = [_synthesize_tool_call(*fb)]

            first = calls[0]
            name = first.function.name
            raw_args = first.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            signature = (name, raw_args)
            if signature == last_signature:
                duplicate_streak += 1
                if duplicate_streak >= AGENT_DUPLICATE_CALL_LIMIT:
                    yield (
                        f"Stopped: model called `{name}` with the same "
                        f"arguments {duplicate_streak + 1} times in a row "
                        f"and isn't making progress."
                    )
                    return
            else:
                duplicate_streak = 0
            last_signature = signature

            yield {"status": self._TOOL_STATUS_MAP.get(name, "Working…")}

            impl = self._impls.get(name)
            result = impl(args) if impl else {"error": f"unknown tool {name}"}

            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": first.id,
                            "type": "function",
                            "function": {"name": name, "arguments": raw_args},
                        }
                    ],
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": first.id, "content": json.dumps(result, default=str)}
            )
        else:
            # Step budget exhausted — stream a tool-free summary.
            yield {"status": "Composing answer…"}
            for event in self.client.chat_stream(
                messages=messages, temperature=0.2, timeout=120,
            ):
                if event.token:
                    yield event.token
            return

        # Stream the final reply (the loop broke because the model had no
        # tool calls).  Re-issue as a streaming call WITHOUT tools so the
        # model produces a clean natural-language reply.  Tokens are yielded
        # raw — scrubbing individual tokens destroys whitespace.
        yield {"status": "Composing answer…"}
        for event in self.client.chat_stream(
            messages=messages, temperature=0.2, timeout=120,
        ):
            if event.token:
                yield event.token


# ---------------------------------------------------------------------------
# External fast-query path (§16.9.2 / §16.9.4)
# ---------------------------------------------------------------------------
#
# External role gets:
#   - top_k FAISS retrieval only (NO with_neighbors expansion — that's the
#     internal "from-graph" superpower per §16.9 trade-off)
#   - chunk-level rerank against the question (task 1, 2026-05-11)
#   - one composer LLM call with thinking=False, streamed raw to the UI
#   - human-readable source titles via display_title (no raw filenames)
#
# The agent loop is bypassed entirely. No tools are exposed to external
# users — the module deliberately doesn't construct a GraphAgent at all
# in external mode (Streamlit calls fast_query_stream directly). Output
# safety relies entirely on EXTERNAL_COMPOSER_SYSTEM_PROMPT and on the
# fact that the prompt never contains internal jargon to begin with
# (display_title strips filenames; no tool definitions enter scope).
# The earlier regex-based scrub and corpus-enumeration filters were
# removed in task 2 (2026-05-11) because they (a) broke markdown
# rendering by buffering tokens at sentence boundaries and (b) defended
# only against LLM hallucinations of RAG-shaped strings that weren't
# real leaks anyway.

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
    "again later or contact an admin. If evidence covers part of the "
    "question and not the rest, answer the covered part and decline the "
    "rest — do not pad.\n\n"
    "Topic-mismatch on follow-ups — judge \"enough information\" against "
    "the conversation context, not just the literal current message. If "
    "the thread has been about the kelp / seaweed industry and the user "
    "asks a deictic follow-up like \"中国呢\" / \"what about China?\" / "
    "\"那美国呢\", they're asking about THAT topic for the new entity "
    "(China's kelp industry, US's kelp industry). If your evidence has "
    "data about China only in a different context (general aquaculture, "
    "capture fisheries) but nothing kelp-specific, that's a topic "
    "mismatch — decline rather than pivot silently to the off-topic "
    "data.\n\n"
    "Persona — you are an analyst, not a system. Don't describe your "
    "knowledge as \"my corpus\" / \"my materials\" / \"my information\" / "
    "\"the documents I have access to\" / \"我的资料\" / \"我的知识库\". "
    "If you must explain a limit, frame it as personal expertise coverage "
    "(\"I don't track that closely\", \"that's outside my coverage area\")."
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
    "  HISTORY: user asks about FAO sustainability initiatives; "
    "assistant lists Blue Transformation.\n"
    "  FOLLOW-UP: \"那欧盟呢？\"\n"
    "  STANDALONE: EU sustainability initiatives in fisheries\n\n"
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


def _rewrite_query_with_history(
    client, question: str, history: list[dict] | None,
) -> str:
    """Ask the LLM to rewrite the user's follow-up as a standalone
    retrieval query, given the conversation so far.

    Replaces the earlier string-concat heuristic. With thinking=False
    the call is ~1-2s and produces semantically richer seeds — e.g.
    "中国呢" with kelp-industry history rewrites to
    "China kelp industry distribution", which embeds far closer to
    the actual corpus content than naive concatenation.

    Failures (LLM error, suspicious output, empty result) fall back
    to the raw question — retrieval will be weaker but the chat
    continues.
    """
    if not history:
        return question
    snapshot_lines: list[str] = []
    for m in history[-10:]:  # cap context — rewrite doesn't need full history
        role = m.get("role")
        content = str(m.get("content", "")).strip()
        if role not in ("user", "assistant") or not content:
            continue
        # Truncate any single turn so a long assistant essay doesn't
        # dominate the rewrite prompt budget.
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
            temperature=0.0,
            thinking=False,  # ~1-2s; rewrite is a low-creativity task
            timeout=30,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
    except Exception:
        return question

    if not rewritten or len(rewritten) > _REWRITE_OUTPUT_MAX_CHARS:
        return question
    # Strip common preamble accidents ("Standalone:", quotes, etc.)
    rewritten = rewritten.lstrip("\"'`「『 ").rstrip("\"'`」』 ")
    for prefix in ("STANDALONE:", "Standalone:", "Query:", "查询:", "标准查询:"):
        if rewritten.upper().startswith(prefix.upper()):
            rewritten = rewritten[len(prefix):].strip()
    return rewritten or question


def fast_query(
    instance: GraphInstance,
    question: str,
    *,
    mode: str = "external",
    history: list[dict] | None = None,
) -> str:
    """Single-call retrieval+composer path for external users (§16.9.2).

    No agent loop, no tools, no neighbor expansion. `history` is the
    rolling conversation window (already token-bounded by the caller via
    `_windowed_history`). Two uses:
      1. Seed enrichment — current question + last user turns are fed
         to the embedding model so follow-ups like "中国呢" still find
         the right nodes.
      2. LLM context — full history is included in the composer call so
         the model can resolve deictic references and follow the
         conversation thread, just like internal mode does.

    `mode` is kept for callers that haven't migrated to the stream path;
    behaviour is identical for `external` and `internal` since the
    output scrub was removed in task 2 (2026-05-11).
    """
    storage = instance.storage
    client = get_client("backend")
    # LLM-based query rewrite (§16 retrieval seed C-option, 2026-05-06):
    # ask the model to fold history + follow-up into a standalone query.
    # ~1-2s with thinking=False; falls back to raw question on failure.
    seed_query = _rewrite_query_with_history(client, question, history)
    # Task 1: widen entity recall to k=10 so the chunk reranker has a
    # bigger pool to choose from. Chunks themselves are scored against
    # `seed_query` inside `_build_chunk_evidence`.
    seeds = top_k(instance.vector_store, storage, seed_query, k=10)
    ctx, included_ids = _build_chunk_evidence(
        instance, seeds,
        question=seed_query,
        include_neighbors=False,
        use_display_title=True,
    )

    # Reinforce traversal log even for external — these are real queries
    # that should bump weights on touched nodes/edges in the next pass.
    touched_edge_ids = [
        e.id for e in storage.edges()
        if e.source_id in included_ids and e.target_id in included_ids
    ]
    append_query(
        storage,
        question=question,
        seed_node_ids=[n.id for n in seeds],
        touched_node_ids=sorted(included_ids),
        touched_edge_ids=touched_edge_ids,
    )

    composer_messages: list[dict] = [
        {"role": "system", "content": EXTERNAL_COMPOSER_SYSTEM_PROMPT},
    ]
    if history:
        composer_messages.extend(history)
    composer_messages.append(
        {"role": "user", "content": f"EVIDENCE:\n{ctx}\n\nQUESTION: {question}"},
    )

    resp = client.chat(
        messages=composer_messages,
        temperature=0.2,
        thinking=False,  # ~15× speedup for external; accuracy matters less than latency
    )
    return resp.choices[0].message.content or "I don't have information on that."


def fast_query_stream(
    instance: GraphInstance,
    question: str,
    *,
    mode: str = "external",
    history: list[dict] | None = None,
) -> Iterator[str]:
    """Streaming variant of :meth:`fast_query`.

    Retrieval and evidence assembly happen synchronously (fast — no LLM),
    then the composer call streams tokens straight through to the UI.
    Tokens are yielded raw so markdown rendering in the front-end stays
    atomic (sentence-buffered scrubbing was removed in task 2).
    """
    storage = instance.storage
    client = get_client("backend")
    seed_query = _rewrite_query_with_history(client, question, history)
    # Task 1: same widened entity recall + chunk rerank as fast_query.
    seeds = top_k(instance.vector_store, storage, seed_query, k=10)
    ctx, included_ids = _build_chunk_evidence(
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
        storage,
        question=question,
        seed_node_ids=[n.id for n in seeds],
        touched_node_ids=sorted(included_ids),
        touched_edge_ids=touched_edge_ids,
    )

    composer_messages: list[dict] = [
        {"role": "system", "content": EXTERNAL_COMPOSER_SYSTEM_PROMPT},
    ]
    if history:
        composer_messages.extend(history)
    composer_messages.append(
        {"role": "user", "content": f"EVIDENCE:\n{ctx}\n\nQUESTION: {question}"},
    )

    for event in client.chat_stream(
        messages=composer_messages,
        temperature=0.2,
        thinking=False,
    ):
        if event.token:
            yield event.token
