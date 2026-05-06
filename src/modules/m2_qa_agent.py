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
from typing import Any, Callable

from src.graph.instance import GraphInstance
from src.graph.retrieval import top_k, with_neighbors
from src.graph.tokens import (
    INGEST_INPUT_MAX_TOKENS,
    INGEST_PDF_PAGE_OVERLAP,
    count_tokens,
)
from src.graph.traversal_log import append_query
from src.llm.routing import get_client

SYSTEM_PROMPT = (
    "You are an agent managing a knowledge graph. "
    "Use the provided tools to answer the user's request. "
    "Knowledge enters the graph in exactly one way: the user places a "
    "document under data/raw/ and asks you to ingest it, in which case "
    "you call `ingest_file`. You have NO ability to write nodes or edges "
    "directly from chat; do not pretend otherwise. "
    "If the user asks what files are available to ingest, call "
    "`list_raw_files` — do not claim you cannot see the directory. "
    "If a tool fits the request, call it. "
    "If the user is just chatting and no tool is needed, reply in natural "
    "language without calling any tool."
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
        seeds = top_k(storage, q, k=5)
        nodes = with_neighbors(storage, seeds)
        # Budget summaries only, ≤20k tiktoken (§12.5).
        ctx_parts: list[str] = []
        tok = 0
        included_ids: set[str] = set()
        for n in nodes:
            line = f"- [{n.type}] {n.label}: {n.summary}"
            t = count_tokens(line)
            if tok + t > RETRIEVAL_BUDGET_TOKENS:
                break
            ctx_parts.append(line)
            tok += t
            included_ids.add(n.id)
        ctx = "\n".join(ctx_parts) or "(empty graph)"
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
                {"role": "system", "content": "Answer the question from the graph context. Cite node labels in square brackets. If insufficient info, say so."},
                {"role": "user", "content": f"GRAPH CONTEXT:\n{ctx}\n\nQUESTION: {q}"},
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
                return {"error": f"no such file: data/raw/{basename}"}

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
