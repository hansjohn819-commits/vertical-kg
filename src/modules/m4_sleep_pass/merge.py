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
from collections import defaultdict
from uuid import uuid4

import numpy as np

from src.graph.instance import GraphInstance
from src.graph.models import DerivedProv, Edge, Node, NodeRef
from src.graph.retrieval import encode_node
from src.graph.storage import GraphStorage
from src.graph.tokens import DETAIL_MAX_TOKENS, SUMMARY_MAX_TOKENS, count_tokens
from src.graph.vector_store import VectorStore
from src.llm.routing import get_client

from .pass_log import log_event
from .state import (
    MERGE_COS_FLOOR,
    MERGE_COS_HIGH,
    MERGE_JACCARD_MIN,
    MERGE_MAX_ITER,
    MERGE_PAIR_CAP_PER_ROUND,
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


def _cleanup_prior_ghosts(
    storage: GraphStorage, vector_store: VectorStore, *, pass_id: str,
) -> int:
    """Delete nodes carrying merged_into from a prior pass (§5.5 one-pass delay).

    Vector-store removal is idempotent — the merge that produced the ghost
    should have already removed it from the index, this is the safety net.
    """
    ghosts = [n for n in storage.nodes() if n.merged_into is not None]
    removed = 0
    for g in ghosts:
        # Only drop ghosts with no incident edges (edges should have been
        # migrated during the merge that created them).
        if not storage.incident_edges(g.id):
            # §16.8.1: capture identity before the node disappears so the
            # log is a self-contained tombstone (label/type/summary_head).
            log_event({
                "kind": "prune_node",
                "pass_id": pass_id,
                "summary": f"removed ghost {g.label}",
                "node_id": g.id,
                "label": g.label,
                "type": g.type,
                "summary_head": (g.summary or "")[:100],
                "merged_into": g.merged_into,
            })
            vector_store.remove(g.id)  # idempotent
            storage.remove_node(g.id)
            removed += 1
    return removed


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _candidate_pairs(
    storage: GraphStorage, vector_store: VectorStore,
) -> list[tuple[Node, Node, float]]:
    """Return deduped (A, B, cos) candidate pairs from two paths.

    Path 1 — FAISS top-K + filter (§16.10): each active node queries the
    index for MERGE_TOP_K+1 nearest neighbors (+1 to drop self), then
    filtered by cos/jac signals. Catches near-duplicate names AND
    moderate-similarity pairs with structural confirmation.

    Path 2 — same-label forced inclusion (2026-05-06 追加): nodes whose
    `label.casefold().strip()` are equal go into the candidate pool
    regardless of where they sit in vector space. Empirical reason:
    same-label duplicates in this corpus often have cos 0.6-0.8 (different
    docs describe the entity with different summary text) and jac=0
    (graph not yet built up shared neighbors), so the FAISS top-K path
    alone misses them. Same-label is itself a near-definitive identity
    signal — cheaper to let LLM judge a few extra borderline cases (like
    distinct people sharing a name) than to re-design the candidate
    generation around acronyms / aliases.
    """
    nodes = _active_nodes(storage)
    if len(nodes) < 2:
        return []

    neighbor_ids: dict[str, set[str]] = {
        n.id: {nb.id for nb in storage.neighbors(n.id)} for n in nodes
    }
    by_id: dict[str, Node] = {n.id: n for n in nodes}

    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[Node, Node, float]] = []
    same_label_added = 0

    # Path 1: FAISS top-K with cos/jac filter.
    for a in nodes:
        # +1 because the index contains `a` itself; self will be top-1 with
        # cos=1 and we filter it out below.
        hits = vector_store.query(encode_node(a), MERGE_TOP_K + 1)
        for hit_id, cos in hits:
            if hit_id == a.id:
                continue
            b = by_id.get(hit_id)
            if b is None:  # vector store stale relative to active set
                continue
            key = tuple(sorted([a.id, b.id]))
            if key in seen:
                continue
            jac = _jaccard(neighbor_ids[a.id], neighbor_ids[b.id])
            # Branch 1: very high cos alone is sufficient (near-duplicate names).
            # Branch 2: moderate cos AND structural confirmation (shared neighbors).
            if cos >= MERGE_COS_HIGH or (cos >= MERGE_COS_FLOOR and jac >= MERGE_JACCARD_MIN):
                seen.add(key)
                pairs.append((a, b, cos))

    # Path 2: same-label forced inclusion. Encode each node once, group by
    # normalized label, compute pairwise cos for in-group pairs (used for
    # sorting / cap; NOT for filtering — same-label is the qualifier here).
    label_groups: dict[str, list[tuple[Node, np.ndarray]]] = defaultdict(list)
    for n in nodes:
        norm = n.label.strip().casefold()
        if not norm:
            continue
        label_groups[norm].append((n, encode_node(n)))
    for group in label_groups.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, va = group[i]
                b, vb = group[j]
                key = tuple(sorted([a.id, b.id]))
                if key in seen:
                    continue
                cos = float(np.dot(va, vb))  # both already normalized
                seen.add(key)
                pairs.append((a, b, cos))
                same_label_added += 1

    if same_label_added:
        log_event({
            "kind": "merge_same_label_added",
            "summary": f"forced {same_label_added} same-label pairs into candidate pool",
            "count": same_label_added,
        })

    # §16.11 hard cap: prevent LLM-judge call count from blowing up on
    # large graphs. After the OR-filter we rank by cos desc and keep the
    # top-N; whatever's clipped lands in log.md so cap value is tunable
    # against real data.
    if len(pairs) > MERGE_PAIR_CAP_PER_ROUND:
        pairs.sort(key=lambda p: p[2], reverse=True)
        clipped = pairs[MERGE_PAIR_CAP_PER_ROUND:]
        pairs = pairs[:MERGE_PAIR_CAP_PER_ROUND]
        log_event({
            "kind": "merge_pair_capped",
            "summary": (
                f"capped {len(clipped)} candidate pairs "
                f"(kept {MERGE_PAIR_CAP_PER_ROUND})"
            ),
            "kept": MERGE_PAIR_CAP_PER_ROUND,
            "dropped": len(clipped),
            "min_cos_kept": pairs[-1][2],
            "max_cos_dropped": clipped[0][2],
        })
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
    """Merge two nodes' descriptions into one. Runs with thinking=False
    (§16.17.7): combining two existing summaries into one is a text-merge
    task, not a reasoning task — the same reason M1 intra-doc fuse is
    thinking-off. The judgement of "are these the same entity?" already
    happened upstream in `_judge_pair` (which IS thinking-on)."""
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
        thinking=False,
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
    vector_store: VectorStore,
    a: Node,
    b: Node,
    fused_summary: str,
    fused_detail: str,
    pass_id: str,
) -> Node:
    # Pick the higher-weighted original's type/label as the canonical.
    primary, secondary = (a, b) if a.weight >= b.weight else (b, a)

    # §16.17 source-text linkage: the fused node inherits chunk pointers
    # from both originals so query-time chunk lookup keeps working after
    # cross-document merges (Tesla / Tesla Inc → Tesla still points back to
    # every chunk that mentioned either form). list(set(...)) preserves
    # uniqueness without enforcing an order — chunks_for_seed() at query
    # time picks an ordering anyway.
    fused_text_unit_ids = list(set(a.text_unit_ids) | set(b.text_unit_ids))

    new_node = Node(
        id=str(uuid4()),
        type=primary.type,
        label=primary.label,
        summary=fused_summary,
        detail=fused_detail,
        weight=a.weight + b.weight,
        version=max(a.version, b.version) + 1,
        provenance=DerivedProv(
            operation_id=f"{pass_id}:merge:{primary.label}+{secondary.label}",
            operation_type="merge",
            inputs=[NodeRef(id=a.id, version=a.version), NodeRef(id=b.id, version=b.version)],
            llm_run_id=pass_id,
        ),
        text_unit_ids=fused_text_unit_ids,
    )
    storage.add_node(new_node)
    # Index the fused node; drop originals from the index immediately so
    # the next round / next chat retrieval can't return stale A or B.
    # Storage still holds A and B (with merged_into set) one pass for
    # provenance — that's a storage-level concern, not vector-level.
    vector_store.add(new_node.id, encode_node(new_node))
    vector_store.remove(a.id)
    vector_store.remove(b.id)

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
            # §16.17 fix (2026-05-10): twin absorbs e — sum weight AND
            # union text_unit_ids so the source-text linkage from the
            # absorbed edge isn't dropped. Same union semantics that
            # _execute_merge applies to the fused node above. Also keep
            # the longer evidence_quote (more informative span) for the
            # downstream show_provenance / chat citation paths.
            twin.weight += e.weight
            if e.text_unit_ids:
                twin.text_unit_ids = sorted(set(twin.text_unit_ids) | set(e.text_unit_ids))
            if e.evidence_quote and len(e.evidence_quote) > len(twin.evidence_quote or ""):
                twin.evidence_quote = e.evidence_quote
            continue
        new_edge = e.model_copy(update={"source_id": new_src, "target_id": new_tgt})
        storage.add_edge(new_edge)

    # Mark originals for one-pass-delayed cleanup (§5.5).
    a.merged_into = new_node.id
    b.merged_into = new_node.id

    # Flatten ghost chains: any prior ghost that pointed at a or b now
    # points at new_node. Keeps depth-of-chain at 1 so show_provenance
    # and any merged_into walker doesn't have to recurse mid-pass.
    # (Without this, A→B→C chains form when a previous-iter ghost's target
    # gets merged again in a later iter of the same pass.)
    for n in storage.nodes():
        if n.merged_into in (a.id, b.id):
            n.merged_into = new_node.id

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
    vector_store = instance.vector_store
    client = get_client("backend")
    pass_id = state.get("pass_id", "unknown")
    iter_idx = int(state.get("merge_iter", 0))

    if iter_idx == 0:
        ghosts_removed = _cleanup_prior_ghosts(storage, vector_store, pass_id=pass_id)
    else:
        ghosts_removed = 0

    pairs = _candidate_pairs(storage, vector_store)
    merged_ids: list[str] = []
    verdicts = {"same": 0, "different": 0, "same_with_caveats": 0, "error": 0}

    # Carry per-pass memory of already-rejected pairs. A pair the judge
    # said "different" about won't change its mind on the same inputs;
    # skipping shortens later rounds from "re-judge everything" to
    # "judge only what the merge state actually changed". Without this
    # an N-round pass redoes the same ~200 LLM calls each round when
    # nothing merges (observed: 2 × 18 min = 36 min wasted, 2026-05-17).
    already_rejected = set(state.get("merge_rejected_pairs") or [])
    new_rejections: list[str] = []
    skipped_already_rejected = 0

    for a, b in [(p[0], p[1]) for p in pairs]:
        # Skip if either side got merged earlier in this same round.
        if a.merged_into is not None or b.merged_into is not None:
            continue
        pair_key = "|".join(sorted([a.id, b.id]))
        if pair_key in already_rejected:
            skipped_already_rejected += 1
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
            new_node = _execute_merge(
                storage, vector_store, a, b, fused_summary, fused_detail, pass_id,
            )
            merged_ids.append(new_node.id)
            # §16.8.1: log the labels and the fused id so a later
            # `list_recent_merges` query can answer "what got merged" even
            # after the originals are ghost-cleaned next pass.
            survivor, retired = (a, b) if a.weight >= b.weight else (b, a)
            log_event({
                "kind": "merge",
                "pass_id": pass_id,
                "summary": f"{a.label} + {b.label} -> {new_node.label}",
                "new_id": new_node.id,
                "fused_label": new_node.label,
                "survivor_id": survivor.id,
                "survivor_label": survivor.label,
                "retired_id": retired.id,
                "retired_label": retired.label,
                "inputs": [a.id, b.id],
                "why": judgement.get("why", ""),
                "evidence": judgement.get("why", ""),
            })
        elif verdict == "same_with_caveats":
            a.weight *= 0.9
            b.weight *= 0.9
            log_event({"kind": "merge_caveat", "pass_id": pass_id, "summary": f"{a.label} ~ {b.label}", "why": judgement.get("why", "")})
        elif verdict == "different":
            # §16.8.5: log every reject so threshold/judge tuning has data
            # to chew on. Round summary alone only gives aggregate counts.
            new_rejections.append(pair_key)
            log_event({
                "kind": "merge_reject",
                "pass_id": pass_id,
                "summary": f"{a.label} ≠ {b.label}",
                "a_id": a.id, "a_label": a.label, "a_type": a.type,
                "b_id": b.id, "b_label": b.label, "b_type": b.type,
                "why": judgement.get("why", ""),
            })

    # Done vote. Short-circuit: if 0 merges happened AND nothing was
    # skipped via the rejection memory, the candidate pool is identical
    # to next round's pool, so any further round produces the same 0
    # merges. No need to spend an LLM call on `_vote_done` — and more
    # importantly, no need to spend another ~18 min on a duplicate
    # merge round if `_vote_done` happens to vote "continue".
    round_summary = (
        f"Merge round {iter_idx + 1} over {len(pairs)} candidate pairs: "
        f"merged {verdicts.get('same', 0)}, different {verdicts.get('different', 0)}, "
        f"caveat {verdicts.get('same_with_caveats', 0)}, errors {verdicts.get('error', 0)}, "
        f"skipped_already_rejected {skipped_already_rejected}."
    )
    if verdicts.get("same", 0) == 0:
        should_stop = True
    else:
        try:
            should_stop = _vote_done(client, round_summary)
        except Exception:
            should_stop = True  # fail safe: exit loop on vote failure
    # Hard cap fallback handled in graph routing.

    stats = dict(state.get("stats") or {})
    stats["merge_total"] = int(stats.get("merge_total", 0)) + len(merged_ids)
    stats["nodes_pruned_total"] = int(stats.get("nodes_pruned_total", 0)) + ghosts_removed
    stats[f"merge_round_{iter_idx}"] = {
        "merged": len(merged_ids),
        "candidates": len(pairs),
        "skipped_already_rejected": skipped_already_rejected,
        **verdicts,
    }

    log_event({
        "kind": "merge_round_done",
        "pass_id": pass_id,
        "summary": round_summary + f" vote_stop={should_stop}",
    })

    return {
        "merge_iter": iter_idx + 1,
        "merge_done_vote": should_stop,
        "merged_new_ids": merged_ids,
        "merge_rejected_pairs": new_rejections,
        "seeded_for_link": merged_ids,
        "stats": stats,
    }


def merge_should_continue(state: PassState) -> str:
    if state.get("merge_done_vote"):
        return "prune"
    if int(state.get("merge_iter", 0)) >= MERGE_MAX_ITER:
        return "prune"
    return "merge"
