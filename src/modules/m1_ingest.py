"""M1: raw text → initial graph (Phase 4).

v1: single-chunk input only. Input must be ≤ INGEST_INPUT_MAX_TOKENS.
Chunking deferred — see guide §12.5.4 / §12.5.5.
"""

import json
import re
import uuid
from dataclasses import dataclass

from src.graph.models import DirectProv, Edge, Node
from src.graph.storage import GraphStorage
from src.graph.tokens import (
    INGEST_INPUT_MAX_TOKENS,
    SUMMARY_MAX_TOKENS,
    count_tokens,
)
from src.llm.routing import get_client

EXTRACTION_SYSTEM_PROMPT = """You are a knowledge-graph extraction assistant.
Given the RAW TEXT, extract at most 5 entities and at most 5 relations between them.

Output STRICT JSON with exactly this shape:
{
  "nodes": [
    {"type": "<entity type>", "label": "<short name>", "summary": "<concise, ≤40 words>"}
  ],
  "edges": [
    {"source_label": "<label from nodes>", "target_label": "<label from nodes>", "type": "<relation>"}
  ]
}

Rules:
- Every edge's source_label and target_label MUST match a label in "nodes".
- Output ONLY the JSON object, no prose before or after.
"""


@dataclass
class IngestResult:
    run_id: str
    nodes_added: int
    edges_added: int
    edges_skipped: int


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


def ingest_raw_text(
    storage: GraphStorage,
    raw_text: str,
    raw_doc_id: str,
) -> IngestResult:
    tok = count_tokens(raw_text)
    if tok > INGEST_INPUT_MAX_TOKENS:
        raise ValueError(
            f"Raw input is {tok} tiktoken tokens, exceeds M1 v1 cap "
            f"{INGEST_INPUT_MAX_TOKENS}. Chunking not implemented — "
            f"see guide §12.5.4/§12.5.5."
        )

    run_id = f"m1-{uuid.uuid4().hex[:8]}"
    client = get_client("backend")
    resp = client.chat(
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ],
        temperature=0.2,
    )
    data = _parse_json_loose(resp.choices[0].message.content or "")

    label_to_node: dict[str, Node] = {}
    for nd in data.get("nodes", []):
        label = str(nd.get("label", "")).strip()
        if not label:
            continue
        summary = str(nd.get("summary", ""))
        if count_tokens(summary) > SUMMARY_MAX_TOKENS:
            summary = summary[: SUMMARY_MAX_TOKENS * 4]
        node = Node(
            type=str(nd.get("type", "Entity")),
            label=label,
            summary=summary,
            provenance=DirectProv(
                raw_doc_id=raw_doc_id,
                extraction_run_id=run_id,
            ),
        )
        storage.add_node(node)
        label_to_node[label] = node

    edges_added = 0
    edges_skipped = 0
    for ed in data.get("edges", []):
        src = label_to_node.get(str(ed.get("source_label", "")).strip())
        tgt = label_to_node.get(str(ed.get("target_label", "")).strip())
        if not src or not tgt:
            edges_skipped += 1
            continue
        storage.add_edge(
            Edge(
                source_id=src.id,
                target_id=tgt.id,
                type=str(ed.get("type", "RELATED_TO")),
                provenance=DirectProv(
                    raw_doc_id=raw_doc_id,
                    extraction_run_id=run_id,
                ),
            )
        )
        edges_added += 1

    return IngestResult(
        run_id=run_id,
        nodes_added=len(label_to_node),
        edges_added=edges_added,
        edges_skipped=edges_skipped,
    )
