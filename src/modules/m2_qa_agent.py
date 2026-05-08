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
    INGEST_PDF_PAGE_OVERLAP,
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

# Budget for graph_query retrieval context (§12.5.2).
RETRIEVAL_BUDGET_TOKENS = 20_000

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
                "description": "Show the provenance (source) of a specific node or edge given its ID.",
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
                    "(text extracted via pdfplumber; long PDFs are auto-chunked by "
                    "page with 2-page overlap, each chunk ingested as a "
                    "separate raw_doc — the result reports per-chunk counts)."
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
        seeds = top_k(self.instance.vector_store, storage, seed_q, k=5)
        nodes = with_neighbors(storage, seeds)
        # Budget summaries only, ≤20k tiktoken (§12.5).
        ctx_parts: list[str] = []
        tok = 0
        included_ids: set[str] = set()
        for n in nodes:
            # Thread the source document into the evidence line so the
            # answer composer can cite it inline ("according to the FAO
            # SOFIA 2024 report...") instead of reading off node IDs.
            src = ""
            prov = getattr(n, "provenance", None)
            rid = getattr(prov, "raw_doc_id", None) if prov is not None else None
            if rid:
                src = f" (source: {rid})"
            line = f"- [{n.type}] {n.label}: {n.summary}{src}"
            t = count_tokens(line)
            if tok + t > RETRIEVAL_BUDGET_TOKENS:
                break
            ctx_parts.append(line)
            tok += t
            included_ids.add(n.id)
        ctx = "\n".join(ctx_parts) or "(no evidence in working memory yet)"
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
                    "user's question using the evidence below. Each evidence "
                    "item lists the source document it came from. Write in "
                    "natural prose the way an industry analyst would — when "
                    "you draw on a specific report, weave the source into "
                    "the sentence (e.g., \"according to the State of the "
                    "Kelp Industry Report...\"). Drop the .pdf suffix and "
                    "any #pages_... fragment when naming a source.\n\n"
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
        tgt = str(args.get("node_or_edge_id", ""))
        n = self.instance.storage.get_node(tgt)
        if n is not None:
            return {"kind": "node", "id": n.id, "provenance": n.provenance.model_dump()}
        return {"kind": "not_found", "id": tgt}

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

    @staticmethod
    def _plan_pdf_chunks(
        page_tokens: list[int], cap: int, overlap: int,
    ) -> list[tuple[int, int]]:
        """Greedy page-pack into chunks with `overlap`-page tail repeat.

        Returns list of (start, end_exclusive) page indices. A page that
        on its own exceeds `cap` becomes a single-page chunk (M1 will then
        raise — caller surfaces the error per-chunk rather than silently
        truncating).
        """
        n = len(page_tokens)
        chunks: list[tuple[int, int]] = []
        i = 0
        while i < n:
            end = i
            running = 0
            while end < n and (end == i or running + page_tokens[end] <= cap):
                # First page of the chunk goes in unconditionally so a single
                # oversize page doesn't get skipped — M1 will raise on it.
                running += page_tokens[end]
                end += 1
                if end == i + 1 and running > cap:
                    # Single page already over cap; close chunk here.
                    break
            chunks.append((i, end))
            if end >= n:
                break
            i = max(end - overlap, i + 1)
        return chunks

    def _impl_ingest_file(self, args: dict) -> dict:
        """Read data/raw/<basename> and ingest it via M1 (DirectProv).

        Path traversal is prevented by reducing to `basename` and then
        asserting the resolved path sits inside the data/raw/ directory.
        Long PDFs are auto-chunked by page with INGEST_PDF_PAGE_OVERLAP-page
        overlap; each chunk ingests as its own raw_doc.
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
            return self._ingest_single(basename, ext, text)

        if ext in self._RAW_PDF_EXTS:
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

            page_tokens = [count_tokens(p) for p in pages]
            total_tokens = sum(page_tokens)

            if total_tokens <= INGEST_INPUT_MAX_TOKENS:
                text = "\n\n".join(pages).strip()
                return self._ingest_single(basename, ext, text, total_pages=len(pages))

            chunk_ranges = self._plan_pdf_chunks(
                page_tokens, INGEST_INPUT_MAX_TOKENS, INGEST_PDF_PAGE_OVERLAP,
            )
            chunk_results: list[dict] = []
            totals = {"nodes_added": 0, "edges_added": 0, "edges_skipped": 0}
            for start, end in chunk_ranges:
                chunk_text = "\n\n".join(pages[start:end]).strip()
                chunk_doc_id = f"{basename}#pages_{start + 1}_{end}"
                entry: dict = {
                    "raw_doc_id": chunk_doc_id,
                    "page_range": [start + 1, end],
                    "tokens": sum(page_tokens[start:end]),
                }
                try:
                    res = self.instance.ingest(chunk_text, raw_doc_id=chunk_doc_id)
                    entry.update({
                        "status": "ok",
                        "run_id": res.run_id,
                        "nodes_added": res.nodes_added,
                        "edges_added": res.edges_added,
                        "edges_skipped": res.edges_skipped,
                    })
                    if res.extraction_warning:
                        entry["extraction_warning"] = res.extraction_warning
                    if res.finish_reason and res.finish_reason != "stop":
                        entry["finish_reason"] = res.finish_reason
                    totals["nodes_added"] += res.nodes_added
                    totals["edges_added"] += res.edges_added
                    totals["edges_skipped"] += res.edges_skipped
                except Exception as exc:
                    entry.update({"status": "error", "error": str(exc)})
                chunk_results.append(entry)
            self.instance.save()
            ok_count = sum(1 for c in chunk_results if c.get("status") == "ok")
            return {
                "status": "ok" if ok_count == len(chunk_results) else "partial",
                "filename": basename,
                "extension": ext,
                "chunked": True,
                "total_pages": len(pages),
                "total_tokens": total_tokens,
                "chunks_processed": ok_count,
                "chunks_total": len(chunk_results),
                "page_overlap": INGEST_PDF_PAGE_OVERLAP,
                "total_nodes_added": totals["nodes_added"],
                "total_edges_added": totals["edges_added"],
                "total_edges_skipped": totals["edges_skipped"],
                "chunks": chunk_results,
            }

        return {
            "error": (
                f"unsupported file extension {ext!r}; "
                f"supported: {sorted(self._RAW_TEXT_EXTS | self._RAW_PDF_EXTS)}"
            )
        }

    def _ingest_single(
        self, basename: str, ext: str, text: str, *, total_pages: int | None = None,
    ) -> dict:
        try:
            result = self.instance.ingest(text, raw_doc_id=basename)
        except Exception as exc:
            return {"error": f"ingest failed: {exc}"}
        self.instance.save()
        out = {
            "status": "ok",
            "filename": basename,
            "extension": ext,
            "chunked": False,
            "run_id": result.run_id,
            "nodes_added": result.nodes_added,
            "edges_added": result.edges_added,
            "edges_skipped": result.edges_skipped,
        }
        if total_pages is not None:
            out["total_pages"] = total_pages
        if result.extraction_warning:
            out["extraction_warning"] = result.extraction_warning
        if result.finish_reason and result.finish_reason != "stop":
            out["finish_reason"] = result.finish_reason
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
# External fast-query path (§16.9.2 / §16.9.4 / §16.9.5)
# ---------------------------------------------------------------------------
#
# External role gets:
#   - top_k FAISS retrieval only (NO with_neighbors expansion — that's the
#     internal "from-graph" superpower per §16.9 trade-off)
#   - one composer LLM call with thinking=False
#   - human-readable source titles via display_title (no raw filenames)
#   - aggressive output scrub for any path / extension / tool-name /
#     UUID / internal jargon that leaks despite the prompt rules
#
# The agent loop is bypassed entirely. No tools are exposed to external
# users — the module deliberately doesn't construct a GraphAgent at all
# in external mode (Streamlit calls fast_query directly).

EXTERNAL_COMPOSER_SYSTEM_PROMPT = (
    "You are a kelp / seaweed industry analyst answering the user's "
    "question using the evidence below. Each evidence item lists the "
    "source it came from. Write in natural prose the way an industry "
    "analyst would — when you draw on a specific report, weave the "
    "source into the sentence (e.g., \"according to the State of the "
    "Kelp Industry Report\"). \n\n"
    "Hard rules — never violate:\n"
    "1. Do NOT mention file paths, file names, file extensions, internal "
    "IDs (UUIDs, hashes), or any tool / function name. The user does not "
    "have access to any backend tooling.\n"
    "2. When citing a source, use only its human-readable title (already "
    "provided in the evidence as `source: <title>`). Never include `.pdf` "
    "or technical fragments like `#pages_X_Y`.\n"
    "3. If the user asks about ingestion, indexing, maintenance, system "
    "internals, or anything outside of industry knowledge, reply: \"I "
    "don't have information on that.\" Do NOT acknowledge that any such "
    "functionality exists or might exist.\n"
    "4. If the evidence does not contain enough information to answer, "
    "decline briefly in the voice of an expert who simply doesn't track "
    "that topic — e.g., \"I don't have specific data on that\" or \"that's "
    "outside what I follow.\" Then STOP. Do NOT speculate. Do NOT direct "
    "the user to \"ask again later\" or \"contact an admin.\" Critically: "
    "do NOT enumerate what topics, regions, or reports your information "
    "DOES cover — phrasings like \"我所掌握的资料主要集中在...\", \"我的"
    "现有资料覆盖...\", \"my information mainly focuses on...\", \"my "
    "evidence covers X but not Y\" reveal that you are reading from a "
    "curated database. A domain expert just declines and stops.\n"
    "    Topic-mismatch case: when judging \"enough information,\" judge "
    "against what the user is asking IN CONVERSATION CONTEXT, not just "
    "the literal current message. If the conversation has been about the "
    "kelp / seaweed industry and the user asks a deictic follow-up like "
    "\"中国呢\" / \"what about China?\" / \"那美国呢\", they're asking "
    "about THAT TOPIC for the new entity (China's kelp industry, US's "
    "kelp industry). If your evidence has data about China in a different "
    "context (e.g., general aquaculture, capture fisheries) but nothing "
    "kelp-specific, that is a topic mismatch — DECLINE rather than pivot "
    "silently to the off-topic data. The user reads every reply as a "
    "continuation of the same thread and will infer the off-topic answer "
    "is on-topic. Honest decline: \"I don't have specific information on "
    "China's kelp industry\" — and stop.\n"
    "5. You are a person — an analyst — not a system. Never refer to your "
    "knowledge as \"my corpus\", \"my materials\", \"my information\", "
    "\"my evidence\", \"my data\", \"the documents I have access to\", "
    "\"我的资料\", \"我的知识库\", \"现有资料\", or anything similar. "
    "If you must explain a limit, use language an industry analyst would "
    "use about their own expertise (\"I don't track that closely\", "
    "\"that's outside my coverage area\").\n"
    "6. Answer ONLY from the evidence provided below. Do NOT add facts "
    "you happen to know from prior training — not even widely-known ones "
    "(company headquarters, product launches, regulatory dates, "
    "geographic facts about countries / regions). The user has no way "
    "to audit prior-training claims, so they are indistinguishable from "
    "fabrication. Mixing prior-training facts with evidence-grounded "
    "facts in the same answer is a hard violation, even when both are "
    "true: the user reads them as one stream and cannot tell which is "
    "which. If evidence covers part of the question and not the rest, "
    "answer the covered part using evidence and decline the rest "
    "(rule 4) — do NOT pad."
)

# Bare regex patterns for output scrub — final safety net if the LLM
# violates a Hard Rule despite the system prompt.
_SCRUB_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Project paths
    (re.compile(r"\bdata/(raw|production|experiment|m1_failures|snapshots)/?\b", re.I), "(unspecified source)"),
    # File extensions glued to a word (PDF / TXT / MD)
    (re.compile(r"(\w)\.(pdf|txt|md|markdown)\b", re.I), r"\1"),
    # Chunk fragment
    (re.compile(r"#pages_\d+_\d+", re.I), ""),
    # Tool / impl names
    (re.compile(
        r"\b(ingest_file|list_raw_files|trigger_sleep_pass|show_provenance|"
        r"mark_stale|read_ontology|graph_query|get_graph_stats|"
        r"list_recent_(?:merges|prunings)|run_plant_recover_eval|"
        r"compare_with_baseline)\b", re.I,
    ), "the system"),
    # UUIDs (8-4-4-4-12 hex)
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), ""),
    # Internal jargon
    (re.compile(r"\b(raw_doc_id|provenance|sleep[ -]?pass|merge candidate|fused node|extraction_run_id)\b", re.I), ""),
]


# Sentences that betray "I'm reading from a curated corpus" — observed
# 2026-05-06 in external chat: when LLM declined ("I don't know about US"),
# it tacked on "我的资料主要集中在中国 / 全球渔业 / 北美海藻产业" which
# IS a corpus enumeration. The rules in EXTERNAL_COMPOSER_SYSTEM_PROMPT
# now forbid this, but Gemma 4 still slips. These regexes match the entire
# offending sentence (until period / Chinese full stop / line break / end)
# and delete it cleanly so the polite decline survives.
_CORPUS_ENUMERATION_PATTERNS: list[re.Pattern] = [
    # Chinese — "我所掌握的资料主要集中在...", "我的现有资料覆盖...",
    # "我能找到的信息主要是...", "我的知识库..."
    # Sentence boundary uses Chinese full stop 。 / 、(no, 、 is comma)
    # / newline ONLY — embedded English abbreviation periods like "Inc."
    # must NOT terminate the match. Comma 。 is NOT 。 (comma vs period).
    re.compile(
        r"我[的所]?(掌握|现有|可获取|拥有|能找到|目前)的?(资料|信息|数据|知识|文献|文档|材料|内容)"
        r"[^。\n]*[。\n]?",
    ),
    re.compile(r"我的(知识库|资料库|信息库|数据库)[^。\n]*[。\n]?"),
    re.compile(r"现有(的)?(资料|信息|数据)[^。\n]*[。\n]?"),
    # English — "my materials/corpus/information/evidence/data
    # mainly|primarily|focus|cover|are about ..."
    # English sentence terminator: . ! ? followed by whitespace/EOL, OR
    # newline. Bare "." inside abbreviations like "U.S." won't match
    # because they're not followed by whitespace.
    re.compile(
        r"\b(my|the)\s+(materials?|corpus|information|knowledge\s*base|"
        r"evidence|data|documents?|sources?|records?)\s+"
        r"(mainly|primarily|chiefly|cover|covers|focus(?:es)?|"
        r"are\s+about|is\s+about|center|centers?|"
        r"contain|contains|include|includes)"
        r"[^.!?\n]*(?:[.!?](?=\s|\Z)|\n|\Z)",
        re.I,
    ),
    # English — "the materials I have access to mainly focus on..."
    re.compile(
        r"\bthe\s+(materials?|information|evidence|data|documents?)\s+"
        r"I\s+(have|can)\s+access(\s+to)?[^.!?\n]*(?:[.!?](?=\s|\Z)|\n|\Z)",
        re.I,
    ),
    # English — "what I have access to ..."
    re.compile(
        r"\bwhat\s+I\s+(have|can)\s+(access|find|see|tell)"
        r"[^.!?\n]*(?:[.!?](?=\s|\Z)|\n|\Z)",
        re.I,
    ),
]


def _strip_corpus_enumerations(text: str) -> str:
    """Remove sentences that enumerate corpus contents. Order-dependent
    with the main scrub: this runs first so the regex sees the original
    LLM output (path / tool-name scrubs would otherwise damage the
    trigger phrases before this runs)."""
    cleaned = text or ""
    for pattern in _CORPUS_ENUMERATION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


def _scrub_external_output(text: str) -> str:
    """Apply the safety-net scrub to text destined for an external user.

    Only run on external mode — internal users see the raw output for
    debugging purposes. Multiple-substitution loop handles cases where
    the first pass leaves residue that a later pattern cleans up.

    Order matters: corpus-enumeration patterns run first because they
    target full sentences that other (token-level) scrubs would damage.
    """
    cleaned = text or ""
    cleaned = _strip_corpus_enumerations(cleaned)
    for pattern, replacement in _SCRUB_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    # Collapse any runs of whitespace introduced by removals.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" +([,.;:!?])", r"\1", cleaned)
    # Empty paragraphs left behind by sentence deletion.
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


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

    Returns the answer text scrubbed for external safety.
    """
    storage = instance.storage
    client = get_client("backend")
    # LLM-based query rewrite (§16 retrieval seed C-option, 2026-05-06):
    # ask the model to fold history + follow-up into a standalone query.
    # ~1-2s with thinking=False; falls back to raw question on failure.
    seed_query = _rewrite_query_with_history(client, question, history)
    seeds = top_k(instance.vector_store, storage, seed_query, k=5)
    # External path INTENTIONALLY omits with_neighbors — vector retrieval
    # only, no graph traversal. See §16.9.2 design table.

    ctx_parts: list[str] = []
    tok = 0
    included_ids: set[str] = set()
    for n in seeds:
        prov = getattr(n, "provenance", None)
        rid = getattr(prov, "raw_doc_id", None) if prov is not None else None
        # Use human title in evidence so the LLM never sees the raw filename.
        title = display_title(rid) if rid else None
        src = f" (source: {title})" if title else ""
        line = f"- [{n.type}] {n.label}: {n.summary}{src}"
        t = count_tokens(line)
        if tok + t > RETRIEVAL_BUDGET_TOKENS:
            break
        ctx_parts.append(line)
        tok += t
        included_ids.add(n.id)
    ctx = "\n".join(ctx_parts) or "(no evidence)"

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
    raw = resp.choices[0].message.content or ""

    if mode == "external":
        return _scrub_external_output(raw) or "I don't have information on that."
    return raw


# ---------------------------------------------------------------------------
# Buffered streaming scrub (§16.15.3)
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s|(?<=\n)")

_FLUSH_BUFFER_LIMIT = 200


def _buffered_scrub_stream(token_iter: Iterator[str]) -> Iterator[str]:
    """Scrub streaming tokens through ``_scrub_external_output`` at sentence
    boundaries.  Yields cleaned sentence chunks as they complete.

    Tokens accumulate in a buffer.  When a sentence boundary is detected the
    completed portion is scrubbed and yielded.  A hard limit flushes the
    buffer even without a boundary (unlikely but safe).
    """
    buf = ""
    for token in token_iter:
        buf += token
        while True:
            m = _SENTENCE_BOUNDARY.search(buf)
            if m is None:
                if len(buf) >= _FLUSH_BUFFER_LIMIT:
                    cleaned = _scrub_external_output(buf)
                    if cleaned:
                        yield cleaned
                    buf = ""
                break
            sentence = buf[: m.end()]
            buf = buf[m.end() :]
            cleaned = _scrub_external_output(sentence)
            if cleaned:
                yield cleaned
    if buf:
        cleaned = _scrub_external_output(buf)
        if cleaned:
            yield cleaned


def fast_query_stream(
    instance: GraphInstance,
    question: str,
    *,
    mode: str = "external",
    history: list[dict] | None = None,
) -> Iterator[str]:
    """Streaming variant of :meth:`fast_query` (§16.15.3).

    Retrieval and evidence assembly happen synchronously (fast — no LLM),
    then the composer call streams through the buffered scrub.
    """
    storage = instance.storage
    client = get_client("backend")
    seed_query = _rewrite_query_with_history(client, question, history)
    seeds = top_k(instance.vector_store, storage, seed_query, k=5)

    ctx_parts: list[str] = []
    tok = 0
    included_ids: set[str] = set()
    for n in seeds:
        prov = getattr(n, "provenance", None)
        rid = getattr(prov, "raw_doc_id", None) if prov is not None else None
        title = display_title(rid) if rid else None
        src = f" (source: {title})" if title else ""
        line = f"- [{n.type}] {n.label}: {n.summary}{src}"
        t = count_tokens(line)
        if tok + t > RETRIEVAL_BUDGET_TOKENS:
            break
        ctx_parts.append(line)
        tok += t
        included_ids.add(n.id)
    ctx = "\n".join(ctx_parts) or "(no evidence)"

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

    def _raw_tokens() -> Iterator[str]:
        for event in client.chat_stream(
            messages=composer_messages,
            temperature=0.2,
            thinking=False,
        ):
            if event.token:
                yield event.token

    if mode == "external":
        yield from _buffered_scrub_stream(_raw_tokens())
    else:
        yield from _raw_tokens()
