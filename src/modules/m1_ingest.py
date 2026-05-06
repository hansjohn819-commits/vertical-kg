"""M1: raw text → initial graph (Phase 4).

v1: single-chunk input only. Input must be ≤ INGEST_INPUT_MAX_TOKENS.
Chunking deferred — see guide §12.5.4 / §12.5.5.

Failure handling: when the extractor LLM returns content that isn't a
parseable JSON object, the raw response is dumped to
`data/m1_failures/<ts>_<docid>_attempt<N>.txt` and we retry once with a
stronger reformat instruction. The final IngestResult carries
`extraction_warning` so the caller (and the chat agent) can distinguish
"document had nothing to extract" from "extraction itself failed".
"""

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.graph.models import DirectProv, Edge, Node
from src.graph.storage import GraphStorage
from src.graph.tokens import (
    INGEST_INPUT_MAX_TOKENS,
    SUMMARY_MAX_TOKENS,
    count_tokens,
)
from src.llm.routing import get_client

EXTRACTION_SYSTEM_PROMPT = """You are a knowledge-graph extraction assistant.
Given the RAW TEXT, extract every distinct entity and every relation
between them that the text supports. Do not skip facts to stay under
an arbitrary count — the goal is faithful coverage of the text.

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
- Extract entities and relations that are explicitly stated or directly
  implied by the text. Do not invent facts.
- Each entity should appear once in "nodes"; reuse its label across edges.
- Every edge's source_label and target_label MUST match a label in "nodes".
- Output ONLY the JSON object, no prose before or after.
"""

# Stronger nudge used on the retry attempt after a parse failure.
EXTRACTION_RETRY_REMINDER = (
    "Your previous response could not be parsed as JSON. "
    "Output ONLY the JSON object — no markdown code fences, no prose, "
    "no explanation, no chain of thought. Start with `{` and end with `}`."
)

FAILURE_DUMP_DIR = Path("data") / "m1_failures"


@dataclass
class IngestResult:
    run_id: str
    nodes_added: int
    edges_added: int
    edges_skipped: int
    # None on clean run, otherwise a short tag like "parse_failed_after_retry"
    # or "parse_failed_recovered_on_retry" so the caller can surface it.
    extraction_warning: str | None = None
    finish_reason: str | None = None


def _parse_json_loose(text: str) -> dict | None:
    """Return parsed JSON dict, or None if no JSON object could be parsed.

    The previous version silently returned `{"nodes":[],"edges":[]}` on
    failure, which is indistinguishable from a successful empty extract —
    callers couldn't tell extraction was broken. None forces the caller
    to handle the failure path explicitly.
    """
    if not text:
        return None
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
    return None


def _dump_failure(
    *, raw_doc_id: str, run_id: str, attempt: int,
    input_tokens: int, finish_reason: str | None, raw_response: str,
) -> Path:
    FAILURE_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = re.sub(r"[^A-Za-z0-9_.\-#]", "_", raw_doc_id)[:80]
    out = FAILURE_DUMP_DIR / f"{ts}_{safe}_attempt{attempt}.txt"
    out.write_text(
        f"=== M1 extraction failure ===\n"
        f"timestamp: {ts}\n"
        f"raw_doc_id: {raw_doc_id}\n"
        f"run_id: {run_id}\n"
        f"attempt: {attempt}\n"
        f"input_tokens: {input_tokens}\n"
        f"output_tokens (estimated): {count_tokens(raw_response)}\n"
        f"finish_reason: {finish_reason}\n"
        f"\n=== Raw LLM response ===\n{raw_response}\n",
        encoding="utf-8",
    )
    return out


def _extract_with_retry(
    client, raw_text: str, raw_doc_id: str, run_id: str, input_tokens: int,
) -> tuple[dict | None, str | None, str | None]:
    """Call the extractor; on parse failure, dump + retry once.

    Returns (parsed_data_or_None, extraction_warning, last_finish_reason).
    """
    base_messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": raw_text},
    ]
    resp = client.chat(messages=base_messages, temperature=0.2)
    content = resp.choices[0].message.content or ""
    finish = getattr(resp.choices[0], "finish_reason", None)
    data = _parse_json_loose(content)
    if data is not None:
        return data, None, finish

    _dump_failure(
        raw_doc_id=raw_doc_id, run_id=run_id, attempt=1,
        input_tokens=input_tokens, finish_reason=finish, raw_response=content,
    )
    # Retry: append the broken response + a stronger reformat instruction.
    retry_messages = base_messages + [
        {"role": "assistant", "content": content},
        {"role": "user", "content": EXTRACTION_RETRY_REMINDER},
    ]
    resp2 = client.chat(messages=retry_messages, temperature=0.4)
    content2 = resp2.choices[0].message.content or ""
    finish2 = getattr(resp2.choices[0], "finish_reason", None)
    data2 = _parse_json_loose(content2)
    if data2 is not None:
        return data2, "parse_failed_recovered_on_retry", finish2

    _dump_failure(
        raw_doc_id=raw_doc_id, run_id=run_id, attempt=2,
        input_tokens=input_tokens, finish_reason=finish2, raw_response=content2,
    )
    return None, "parse_failed_after_retry", finish2


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
    data, warning, finish = _extract_with_retry(
        client, raw_text, raw_doc_id, run_id, tok,
    )
    if data is None:
        # Both attempts failed; nothing to add. Caller sees warning.
        return IngestResult(
            run_id=run_id,
            nodes_added=0,
            edges_added=0,
            edges_skipped=0,
            extraction_warning=warning,
            finish_reason=finish,
        )

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
        extraction_warning=warning,
        finish_reason=finish,
    )
