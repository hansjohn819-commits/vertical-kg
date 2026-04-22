"""M2: Agent + tool box (Phase 5).

Full §6.2 tool set. Agent is a simple single-step OpenAI tool loop —
LangGraph is reserved for Phase 6 sleep-pass orchestration (§5.0). Tools
that depend on later phases return stub payloads so the routing test can
still see them in the schema list.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.graph.instance import GraphInstance
from src.graph.retrieval import top_k, with_neighbors
from src.graph.tokens import count_tokens
from src.graph.traversal_log import append_query
from src.llm.routing import get_client
from src.modules.m3_integrate import UserContext, upsert_edge, upsert_node

SYSTEM_PROMPT = (
    "You are an agent managing a knowledge graph. "
    "Use the provided tools to answer the user's request. "
    "If a tool fits the request, call it. "
    "If the user is just chatting and no tool is needed, reply in natural "
    "language without calling any tool."
)

# Budget for graph_query retrieval context (§12.5.2).
RETRIEVAL_BUDGET_TOKENS = 20_000

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
        "upsert_node": {
            "type": "function",
            "function": {
                "name": "upsert_node",
                "description": "Create a new node in the graph (or return the existing one if type+label match). Use when the user explicitly asks to create, add, or record a new entity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "description": "Entity type (e.g., Company, Person)"},
                        "label": {"type": "string", "description": "Short display name for the entity"},
                        "summary": {"type": "string", "description": "Brief description of the entity (≤40 words)"},
                    },
                    "required": ["type", "label"],
                },
            },
        },
        "upsert_edge": {
            "type": "function",
            "function": {
                "name": "upsert_edge",
                "description": "Create an edge between two existing nodes. Use when the user explicitly asks to link, connect, or relate two entities.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_label": {"type": "string", "description": "Label of the source node"},
                        "target_label": {"type": "string", "description": "Label of the target node"},
                        "type": {"type": "string", "description": "Relation type (e.g., CEO_OF, LOCATED_IN)"},
                    },
                    "required": ["source_label", "target_label", "type"],
                },
            },
        },
        "ingest_file": {
            "type": "function",
            "function": {
                "name": "ingest_file",
                "description": (
                    "Load an authoritative raw document from data/raw/ and "
                    "ingest it via M1 (DirectProv). Use when the user asks "
                    "to load / import / ingest a file or document by name. "
                    "NOT for user-contributed facts pasted into chat — "
                    "those are M3's job via the Streamlit Ingest mode."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "File basename under data/raw/ (e.g., 'q1_report.txt'). Path components are stripped for safety.",
                        },
                    },
                    "required": ["filename"],
                },
            },
        },
    }


ALL_TOOL_NAMES = list(_tool_schemas().keys())
DEFAULT_ROUND1_TOOLS = [n for n in ALL_TOOL_NAMES if n not in ("upsert_node", "upsert_edge")]
DEFAULT_ROUND2_TOOLS = list(ALL_TOOL_NAMES)


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
        user_ctx: UserContext | None = None,
    ):
        self.instance = instance
        self.user_ctx = user_ctx or UserContext(
            user_id="demo-user", conversation_id="demo-conv", turn_id="t0",
        )
        self._all_schemas = _tool_schemas()
        names = expose_tools if expose_tools is not None else DEFAULT_ROUND2_TOOLS
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
            "upsert_node": self._impl_upsert_node,
            "upsert_edge": self._impl_upsert_edge,
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

    def _impl_upsert_node(self, args: dict) -> dict:
        n = upsert_node(
            self.instance.storage,
            type=str(args.get("type", "Entity")),
            label=str(args.get("label", "")),
            summary=str(args.get("summary", "")),
            user_ctx=self.user_ctx,
        )
        return {"id": n.id, "type": n.type, "label": n.label}

    def _impl_upsert_edge(self, args: dict) -> dict:
        src = None
        tgt = None
        src_label = str(args.get("source_label", ""))
        tgt_label = str(args.get("target_label", ""))
        for n in self.instance.storage.nodes():
            if n.label == src_label and src is None:
                src = n
            if n.label == tgt_label and tgt is None:
                tgt = n
        if src is None or tgt is None:
            return {"error": "source or target not found by label", "source_found": src is not None, "target_found": tgt is not None}
        e = upsert_edge(
            self.instance.storage,
            source_id=src.id,
            target_id=tgt.id,
            type=str(args.get("type", "RELATED_TO")),
            user_ctx=self.user_ctx,
        )
        return {"source_id": e.source_id, "target_id": e.target_id, "type": e.type}

    def _impl_ingest_file(self, args: dict) -> dict:
        """Read data/raw/<basename> and ingest it via M1 (DirectProv).

        Path traversal is prevented by reducing to `basename` and then
        asserting the resolved path sits inside the data/raw/ directory.
        """
        raw_arg = str(args.get("filename", "")).strip()
        if not raw_arg:
            return {"error": "filename is required"}
        basename = Path(raw_arg).name  # strip any directory components
        raw_dir = (Path("data") / "raw").resolve()
        target = (raw_dir / basename).resolve()
        try:
            target.relative_to(raw_dir)
        except ValueError:
            return {"error": f"refused: path escapes data/raw/ ({basename!r})"}
        if not target.exists() or not target.is_file():
            return {"error": f"no such file: data/raw/{basename}"}
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            return {"error": f"could not read as utf-8: {exc}"}
        try:
            result = self.instance.ingest(text, raw_doc_id=basename)
        except Exception as exc:
            return {"error": f"ingest failed: {exc}"}
        self.instance.save()
        return {
            "status": "ok",
            "filename": basename,
            "run_id": result.run_id,
            "nodes_added": result.nodes_added,
            "edges_added": result.edges_added,
            "edges_skipped": result.edges_skipped,
        }

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
                temperature=0.7,
                timeout=120,
            )
        except Exception as exc:
            return RouteResult(None, None, None, None, error=str(exc))
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
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
        """Full loop: route, execute tool if any, synthesize final reply."""
        if self.instance.sleep_pass_running:
            return "Sleep pass is currently running; chat is paused until it finishes."
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        resp = self.client.chat(
            messages=messages, tools=self.schemas, tool_choice="auto", temperature=0.7, timeout=120,
        )
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            return msg.content or ""
        first = calls[0]
        name = first.function.name
        try:
            args = json.loads(first.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        impl = self._impls.get(name)
        result = impl(args) if impl else {"error": f"unknown tool {name}"}
        # Append assistant tool call + tool result, ask for natural-language synthesis.
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": first.id,
                        "type": "function",
                        "function": {"name": name, "arguments": first.function.arguments},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": first.id, "content": json.dumps(result, default=str)})
        final = self.client.chat(messages=messages, temperature=0.2, timeout=120)
        return final.choices[0].message.content or ""
