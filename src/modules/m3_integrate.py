"""M3: online write from conversation (Phase 4).

Exposes `upsert_node` / `upsert_edge` primitives (also the agent tools in
Phase 5) and the high-level `integrate_user_statement` that
LLM-extracts + classifies + writes.

Per guide §7.3: exact (type, label) match reuses an existing node ID.
This is NOT a merge — semantic merge is M4b's job. Exact match is a
trivial, zero-cost dedup that keeps the demo graph from ballooning.
"""

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from src.graph.models import Edge, Node, UserProv
from src.graph.storage import GraphStorage
from src.graph.tokens import SUMMARY_MAX_TOKENS, count_tokens
from src.llm.routing import get_client

Classification = Literal["experience", "supplement"]


class UserContext(BaseModel):
    user_id: str
    conversation_id: str
    turn_id: str


# --- Primitives ---------------------------------------------------------

def upsert_node(
    storage: GraphStorage,
    *,
    type: str,
    label: str,
    summary: str,
    user_ctx: UserContext,
    classification: Classification = "supplement",
) -> Node:
    for existing in storage.nodes():
        if existing.type == type and existing.label == label:
            return existing
    if count_tokens(summary) > SUMMARY_MAX_TOKENS:
        summary = summary[: SUMMARY_MAX_TOKENS * 4]
    node = Node(
        type=type,
        label=label,
        summary=summary,
        provenance=UserProv(
            user_id=user_ctx.user_id,
            conversation_id=user_ctx.conversation_id,
            turn_id=user_ctx.turn_id,
            classification=classification,
        ),
    )
    storage.add_node(node)
    return node


def upsert_edge(
    storage: GraphStorage,
    *,
    source_id: str,
    target_id: str,
    type: str,
    user_ctx: UserContext,
    classification: Classification = "supplement",
) -> Edge:
    edge = Edge(
        source_id=source_id,
        target_id=target_id,
        type=type,
        provenance=UserProv(
            user_id=user_ctx.user_id,
            conversation_id=user_ctx.conversation_id,
            turn_id=user_ctx.turn_id,
            classification=classification,
        ),
    )
    storage.add_edge(edge)
    return edge


# --- Classification -----------------------------------------------------

CLASSIFY_SYSTEM_PROMPT = """Classify the user's statement as ONE word:

- "experience": user links two already-established concepts in a new way
  (e.g., "in my past A project, X and Y turned out to be related").
- "supplement": user introduces a new fact / entity / attribute
  (e.g., "X's CEO is Y").

Output exactly one word: experience or supplement.
"""


def classify_user_fact(statement: str) -> Classification:
    client = get_client("backend")
    resp = client.chat(
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": statement},
        ],
        temperature=0.0,
    )
    out = (resp.choices[0].message.content or "").strip().lower()
    return "experience" if "experience" in out else "supplement"


# --- High-level integration --------------------------------------------

INTEGRATE_SYSTEM_PROMPT = """You are a knowledge-graph integration assistant.
Given the USER STATEMENT, extract the entities and relations the user is
asserting. Output STRICT JSON:

{
  "nodes": [{"type": "...", "label": "...", "summary": "<≤40 words>"}],
  "edges": [{"source_label": "...", "target_label": "...", "type": "..."}]
}

Rules:
- Only include facts the user explicitly asserts, not background common sense.
- Every edge's source_label and target_label must appear in "nodes".
- Output ONLY the JSON object.
"""


@dataclass
class IntegrateResult:
    classification: Classification
    nodes_touched: int
    edges_added: int


def _parse_json_loose(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"nodes": [], "edges": []}


def integrate_user_statement(
    storage: GraphStorage,
    statement: str,
    user_ctx: UserContext,
) -> IntegrateResult:
    classification = classify_user_fact(statement)
    client = get_client("backend")
    resp = client.chat(
        messages=[
            {"role": "system", "content": INTEGRATE_SYSTEM_PROMPT},
            {"role": "user", "content": statement},
        ],
        temperature=0.2,
    )
    data = _parse_json_loose(resp.choices[0].message.content or "")

    label_to_node: dict[str, Node] = {}
    for nd in data.get("nodes", []):
        label = str(nd.get("label", "")).strip()
        if not label:
            continue
        n = upsert_node(
            storage,
            type=str(nd.get("type", "Entity")),
            label=label,
            summary=str(nd.get("summary", "")),
            user_ctx=user_ctx,
            classification=classification,
        )
        label_to_node[label] = n

    edges_added = 0
    for ed in data.get("edges", []):
        src = label_to_node.get(str(ed.get("source_label", "")).strip())
        tgt = label_to_node.get(str(ed.get("target_label", "")).strip())
        if not src or not tgt:
            continue
        upsert_edge(
            storage,
            source_id=src.id,
            target_id=tgt.id,
            type=str(ed.get("type", "RELATED_TO")),
            user_ctx=user_ctx,
            classification=classification,
        )
        edges_added += 1

    return IntegrateResult(
        classification=classification,
        nodes_touched=len(label_to_node),
        edges_added=edges_added,
    )
