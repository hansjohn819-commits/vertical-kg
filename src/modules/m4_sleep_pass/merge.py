"""4b: entity consolidation (merge). Guide §5.5.

One round per invocation:
1. Clean up ghosts left by the previous pass (merged_into != None from an
   earlier run — §5.5 says those linger one pass for provenance).
2. Generate candidate pairs via embedding top-k + Jaccard neighbor overlap.
3. Fresh LLM instance judges each pair: same / different / same_with_caveats.
4. Execute `same` merges — new node, fused summary/detail, edges migrated
   with weight-sum dedup, originals keep merged_into pointer.
5. LLM votes whether to continue (guide §5.0: explicit done vote).

Convergence: LangGraph loops until the LLM votes done or MERGE_MAX_ITER
rounds elapse.
"""

import json
import re
from uuid import uuid4

import numpy as np

from src.graph.instance import GraphInstance
from src.graph.models import DerivedProv, Edge, Node, NodeRef
from src.graph.retrieval import _get_model
from src.graph.storage import GraphStorage
from src.graph.tokens import DETAIL_MAX_TOKENS, SUMMARY_MAX_TOKENS, count_tokens
from src.llm.routing import get_client

from .pass_log import log_event
from .state import (
    MERGE_COS_MIN,
    MERGE_COS_SAME_TYPE,
    MERGE_JACCARD_MIN,
    MERGE_MAX_ITER,
    MERGE_TOP_K,
    PassState,
)

JUDGE_SYSTEM_PROMPT = """You are a knowledge-graph entity-consolidation judge.
You compare two candidate nodes and decide whether they refer to the SAME
real-world entity.

Reply with STRICT JSON only, exactly these keys:
{
  "verdict": "same" | "different" | "same_with_caveats",
  "why": "<one or two sentences>"
}

Use "same" when they clearly refer to one entity (even if names differ).
Use "different" when they are distinct entities (even if names look similar).
Use "same_with_caveats" when the evidence is mixed — they MIGHT be the same
but you need more signal. This lowers their weights and defers the decision
to a later pass.
"""

FUSION_SYSTEM_PROMPT = """You fuse two knowledge-graph nodes that represent
the same entity into one. Reply with STRICT JSON only, these keys:

{
  "summary": "<<=200 words combined summary, canonical label in text>",
  "reconciliation": "<one short paragraph noting any factual inconsistencies
    between the two source nodes and which one appears authoritative>"
}
"""

DONE_VOTE_SYSTEM_PROMPT = """You are overseeing a knowledge-graph merge loop.
Given the round summary, decide whether the merge loop should STOP (all
obvious consolidations done) or CONTINUE (you suspect more real merges are
available if we try another round).

Reply with STRICT JSON only:
{"decision": "stop" | "continue", "why": "<one sentence>"}
"""


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
    return {}


def _active_nodes(storage: GraphStorage) -> list[Node]:
    return [n for n in storage.nodes() if n.merged_into is None]


def _cleanup_prior_ghosts(storage: GraphStorage) -> int:
    """Delete nodes carrying merged_into from a prior pass (§5.5 one-pass delay)."""
    ghosts = [n for n in storage.nodes() if n.merged_into is not None]
    removed = 0
    for g in ghosts:
        # Only drop ghosts with no incident edges (edges should have been
        # migrated during the merge that created them).
        if not storage.incident_edges(g.id):
            storage.remove_node(g.id)
            removed += 1
    return removed


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _candidate_pairs(storage: GraphStorage) -> list[tuple[Node, Node, float]]:
    """Return deduped (A, B, cos) candidate pairs passing top-k + filter."""
    nodes = _active_nodes(storage)
    if len(nodes) < 2:
        return []
    model = _get_model()
    texts = [f"{n.type}: {n.label} — {n.summary}" for n in nodes]
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    sim = embs @ embs.T
    np.fill_diagonal(sim, -1.0)

    neighbor_ids: dict[str, set[str]] = {
        n.id: {nb.id for nb in storage.neighbors(n.id)} for n in nodes
    }

    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[Node, Node, float]] = []
    for i, a in enumerate(nodes):
        top_idx = np.argsort(-sim[i])[: MERGE_TOP_K]
        for j in top_idx:
            if j == i or sim[i, j] < 0:
                continue
            b = nodes[int(j)]
            key = tuple(sorted([a.id, b.id]))
            if key in seen:
                continue
            cos = float(sim[i, j])
            jac = _jaccard(neighbor_ids[a.id], neighbor_ids[b.id])
            same_type_match = a.type == b.type and cos >= MERGE_COS_SAME_TYPE
            if cos >= MERGE_COS_MIN or jac >= MERGE_JACCARD_MIN or same_type_match:
                seen.add(key)
                pairs.append((a, b, cos))
    return pairs


def _format_node_for_judge(storage: GraphStorage, n: Node) -> str:
    nbrs = storage.neighbors(n.id)
    nbr_lines = "\n".join(f"    - [{nb.type}] {nb.label}: {nb.summary[:120]}" for nb in nbrs[:20])
    detail_cap = n.detail[:2000]
    return (
        f"Node id: {n.id}\n"
        f"Type: {n.type}\n"
        f"Label: {n.label}\n"
        f"Summary: {n.summary}\n"
        f"Detail (truncated): {detail_cap}\n"
        f"Neighbors ({len(nbrs)}):\n{nbr_lines or '    (none)'}\n"
    )


def _judge_pair(client, storage: GraphStorage, a: Node, b: Node) -> dict:
    user = (
        "Candidate A:\n" + _format_node_for_judge(storage, a) +
        "\nCandidate B:\n" + _format_node_for_judge(storage, b)
    )
    resp = client.chat(
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    return _parse_json_loose(resp.choices[0].message.content or "")


def _fuse(client, a: Node, b: Node) -> tuple[str, str]:
    user = (
        "Node A:\nLabel: {al}\nSummary: {as_}\nDetail: {ad}\n\n"
        "Node B:\nLabel: {bl}\nSummary: {bs}\nDetail: {bd}"
    ).format(
        al=a.label, as_=a.summary, ad=a.detail[:2000],
        bl=b.label, bs=b.summary, bd=b.detail[:2000],
    )
    resp = client.chat(
        messages=[
            {"role": "system", "content": FUSION_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
    )
    parsed = _parse_json_loose(resp.choices[0].message.content or "")
    summary = str(parsed.get("summary", f"{a.label} / {b.label}"))[: SUMMARY_MAX_TOKENS * 4]
    reconciliation = str(parsed.get("reconciliation", ""))
    combined_detail = (a.detail + "\n\n---\n" + b.detail + "\n\nReconciliation: " + reconciliation)
    if count_tokens(combined_detail) > DETAIL_MAX_TOKENS:
        combined_detail = combined_detail[: DETAIL_MAX_TOKENS * 4]
    return summary, combined_detail


def _execute_merge(
    storage: GraphStorage,
    a: Node,
    b: Node,
    fused_summary: str,
    fused_detail: str,
    pass_id: str,
) -> Node:
    # Pick the higher-weighted original's type/label as the canonical.
    primary, secondary = (a, b) if a.weight >= b.weight else (b, a)

    new_node = Node(
        id=str(uuid4()),
        type=primary.type,
        label=primary.label,
        summary=fused_summary,
        detail=fused_detail,
        weight=a.weight + b.weight,
        provenance=DerivedProv(
            operation_id=f"{pass_id}:merge:{primary.label}+{secondary.label}",
            operation_type="merge",
            inputs=[NodeRef(id=a.id, version=a.version), NodeRef(id=b.id, version=b.version)],
            llm_run_id=pass_id,
        ),
    )
    storage.add_node(new_node)

    # Migrate edges from A and B to new_node, deduping by (src, tgt, type).
    incident = {e.id: e for e in storage.incident_edges(a.id) + storage.incident_edges(b.id)}
    # Build an index of already-present edges on new_node to sum into.
    def _find_twin(src: str, tgt: str, type_: str) -> Edge | None:
        for e in storage.incident_edges(new_node.id):
            if e.source_id == src and e.target_id == tgt and e.type == type_:
                return e
        return None

    for e in incident.values():
        new_src = new_node.id if e.source_id in (a.id, b.id) else e.source_id
        new_tgt = new_node.id if e.target_id in (a.id, b.id) else e.target_id
        storage.remove_edge_by_id(e.id)
        if new_src == new_tgt:
            continue  # drop self-loops produced by the A-B edge itself
        twin = _find_twin(new_src, new_tgt, e.type)
        if twin is not None:
            twin.weight += e.weight
            continue
        new_edge = e.model_copy(update={"source_id": new_src, "target_id": new_tgt})
        storage.add_edge(new_edge)

    # Mark originals for one-pass-delayed cleanup (§5.5).
    a.merged_into = new_node.id
    b.merged_into = new_node.id
    return new_node


def _vote_done(client, round_summary: str) -> bool:
    resp = client.chat(
        messages=[
            {"role": "system", "content": DONE_VOTE_SYSTEM_PROMPT},
            {"role": "user", "content": round_summary},
        ],
        temperature=0.0,
    )
    parsed = _parse_json_loose(resp.choices[0].message.content or "")
    return str(parsed.get("decision", "stop")).lower() == "stop"


def merge_step(state: PassState, *, instance: GraphInstance) -> dict:
    storage = instance.storage
    client = get_client("backend")
    pass_id = state.get("pass_id", "unknown")
    iter_idx = int(state.get("merge_iter", 0))

    if iter_idx == 0:
        _cleanup_prior_ghosts(storage)

    pairs = _candidate_pairs(storage)
    merged_ids: list[str] = []
    verdicts = {"same": 0, "different": 0, "same_with_caveats": 0, "error": 0}

    for a, b in [(p[0], p[1]) for p in pairs]:
        # Skip if either side got merged earlier in this same round.
        if a.merged_into is not None or b.merged_into is not None:
            continue
        try:
            judgement = _judge_pair(client, storage, a, b)
        except Exception as exc:
            verdicts["error"] += 1
            log_event({"kind": "merge_judge_error", "pass_id": pass_id, "a": a.label, "b": b.label, "summary": str(exc)[:120]})
            continue
        verdict = str(judgement.get("verdict", "different")).lower()
        verdicts[verdict] = verdicts.get(verdict, 0) + 1

        if verdict == "same":
            fused_summary, fused_detail = _fuse(client, a, b)
            new_node = _execute_merge(storage, a, b, fused_summary, fused_detail, pass_id)
            merged_ids.append(new_node.id)
            log_event({
                "kind": "merge",
                "pass_id": pass_id,
                "summary": f"{a.label} + {b.label} -> {new_node.label}",
                "new_id": new_node.id,
                "inputs": [a.id, b.id],
                "why": judgement.get("why", ""),
            })
        elif verdict == "same_with_caveats":
            a.weight *= 0.9
            b.weight *= 0.9
            log_event({"kind": "merge_caveat", "pass_id": pass_id, "summary": f"{a.label} ~ {b.label}", "why": judgement.get("why", "")})

    # Done vote (LLM, one call per round).
    round_summary = (
        f"Merge round {iter_idx + 1} over {len(pairs)} candidate pairs: "
        f"merged {verdicts.get('same', 0)}, different {verdicts.get('different', 0)}, "
        f"caveat {verdicts.get('same_with_caveats', 0)}, errors {verdicts.get('error', 0)}."
    )
    try:
        should_stop = _vote_done(client, round_summary)
    except Exception:
        should_stop = True  # fail safe: exit loop on vote failure
    # Hard cap fallback handled in graph routing.

    stats = dict(state.get("stats") or {})
    stats["merge_total"] = int(stats.get("merge_total", 0)) + len(merged_ids)
    stats[f"merge_round_{iter_idx}"] = {"merged": len(merged_ids), "candidates": len(pairs), **verdicts}

    log_event({
        "kind": "merge_round_done",
        "pass_id": pass_id,
        "summary": round_summary + f" vote_stop={should_stop}",
    })

    return {
        "merge_iter": iter_idx + 1,
        "merge_done_vote": should_stop,
        "merged_new_ids": merged_ids,
        "seeded_for_link": merged_ids,
        "stats": stats,
    }


def merge_should_continue(state: PassState) -> str:
    if state.get("merge_done_vote"):
        return "prune"
    if int(state.get("merge_iter", 0)) >= MERGE_MAX_ITER:
        return "prune"
    return "merge"
