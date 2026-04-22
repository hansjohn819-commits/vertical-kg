"""4d: derived-edge formation. Guide §5.7.

Seed nodes = whatever 4b merged this pass (state.seeded_for_link) plus
every node whose weight is in the top decile (a rough "reinforced" proxy
since 4c didn't track deltas). For each seed, BFS to depth LINK_BFS_DEPTH,
and for every (seed, remote) pair with no direct edge a fresh LLM decides
whether a real relation exists. The LLM MUST state both what the relation
is and why — missing either and the pair is dropped (§5.7 key rule).

Convergence: mechanical. Pairs judged in an earlier round are remembered
per-pass so they are not re-asked; when a round adds no new edges it exits.
"""

import json
import re
from collections import deque

from src.graph.instance import GraphInstance
from src.graph.models import DerivedProv, Edge, NodeRef
from src.graph.storage import GraphStorage
from src.llm.routing import get_client

from .pass_log import log_event
from .state import LINK_BFS_DEPTH, LINK_MAX_NEW_PER_ROUND, PassState

LINK_SYSTEM_PROMPT = """You are a knowledge-graph discovery judge. You look
at two nodes (plus the path of intermediate nodes that link them) and decide
whether a DIRECT semantic relation exists between the two endpoints.

Reply with STRICT JSON only:
{
  "holds": true | false,
  "what": "<name of the relation, or empty string if holds=false>",
  "why": "<one or two sentences explaining why the relation holds>",
  "edge_type": "<suggested UPPER_SNAKE_CASE type, or RELATED_TO>"
}

Critical rules:
- If you cannot articulate both WHAT the relation is AND WHY it holds in
  plain language, set holds=false. A topological shortcut is not a relation.
- Do not invent facts not implied by the node summaries and the path.
- Avoid trivial edges (e.g., both are companies -> hardly a relation).
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


def _bfs_pairs(
    storage: GraphStorage,
    seed_id: str,
    depth: int,
) -> list[tuple[str, list[str]]]:
    """Return (remote_id, path) pairs within `depth` hops of seed."""
    if storage.get_node(seed_id) is None:
        return []
    visited: dict[str, list[str]] = {seed_id: [seed_id]}
    q: deque[tuple[str, int]] = deque([(seed_id, 0)])
    while q:
        node_id, d = q.popleft()
        if d == depth:
            continue
        for nb in storage.neighbors(node_id):
            if nb.id in visited:
                continue
            visited[nb.id] = visited[node_id] + [nb.id]
            q.append((nb.id, d + 1))
    return [(nid, path) for nid, path in visited.items() if nid != seed_id]


def _direct_edge_exists(storage: GraphStorage, a: str, b: str) -> bool:
    for e in storage.incident_edges(a):
        if (e.source_id, e.target_id) in ((a, b), (b, a)):
            return True
    return False


def _seeds(state: PassState, storage: GraphStorage) -> list[str]:
    merged = [nid for nid in state.get("seeded_for_link", []) if storage.get_node(nid) is not None]
    # Top-decile reinforced proxy: nodes in the top 10% by weight.
    active = [n for n in storage.nodes() if n.merged_into is None]
    if active:
        active.sort(key=lambda n: n.weight, reverse=True)
        cutoff = max(1, len(active) // 10)
        merged = list({*merged, *(n.id for n in active[:cutoff])})
    return merged


def link_step(state: PassState, *, instance: GraphInstance) -> dict:
    storage = instance.storage
    client = get_client("backend")
    pass_id = state.get("pass_id", "unknown")
    iter_idx = int(state.get("link_iter", 0))

    tried: set[str] = set(state.get("link_tried_pairs") or [])  # sorted-pair strings
    seeds = _seeds(state, storage)

    added = 0
    asked = 0
    for seed_id in seeds:
        for remote_id, path in _bfs_pairs(storage, seed_id, LINK_BFS_DEPTH):
            if added >= LINK_MAX_NEW_PER_ROUND:
                break
            key = "|".join(sorted([seed_id, remote_id]))
            if key in tried:
                continue
            tried.add(key)
            if _direct_edge_exists(storage, seed_id, remote_id):
                continue
            seed_node = storage.get_node(seed_id)
            remote_node = storage.get_node(remote_id)
            if seed_node is None or remote_node is None:
                continue

            path_summary = " -> ".join(
                f"[{storage.get_node(pid).type}] {storage.get_node(pid).label}"  # type: ignore[union-attr]
                for pid in path
                if storage.get_node(pid) is not None
            )
            user = (
                f"Seed node: [{seed_node.type}] {seed_node.label}\n"
                f"  Summary: {seed_node.summary}\n\n"
                f"Remote node: [{remote_node.type}] {remote_node.label}\n"
                f"  Summary: {remote_node.summary}\n\n"
                f"Path: {path_summary}\n"
            )
            asked += 1
            try:
                resp = client.chat(
                    messages=[
                        {"role": "system", "content": LINK_SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.1,
                )
            except Exception as exc:
                log_event({"kind": "link_judge_error", "pass_id": pass_id, "summary": str(exc)[:120]})
                continue
            parsed = _parse_json_loose(resp.choices[0].message.content or "")
            if not parsed.get("holds"):
                continue
            what = str(parsed.get("what", "")).strip()
            why = str(parsed.get("why", "")).strip()
            if not what or not why:
                continue  # strict §5.7: discard pairs without clear what+why

            edge_type = str(parsed.get("edge_type", "RELATED_TO")).strip() or "RELATED_TO"
            new_edge = Edge(
                source_id=seed_id,
                target_id=remote_id,
                type=edge_type,
                weight=0.5,  # new derived edges start mid-weight (§5.4 PageRank-like)
                provenance=DerivedProv(
                    operation_id=f"{pass_id}:link:{seed_node.label}->{remote_node.label}",
                    operation_type="traversal",
                    inputs=[NodeRef(id=pid, version=storage.get_node(pid).version)  # type: ignore[union-attr]
                            for pid in path if storage.get_node(pid) is not None],
                    llm_run_id=pass_id,
                ),
            )
            storage.add_edge(new_edge)
            added += 1
            log_event({
                "kind": "link_form",
                "pass_id": pass_id,
                "summary": f"{seed_node.label} -[{edge_type}]-> {remote_node.label}",
                "what": what,
                "why": why,
            })
        if added >= LINK_MAX_NEW_PER_ROUND:
            break

    stats = dict(state.get("stats") or {})
    stats["link_total"] = int(stats.get("link_total", 0)) + added
    stats[f"link_round_{iter_idx}"] = {"asked": asked, "added": added}

    log_event({
        "kind": "link_round_done",
        "pass_id": pass_id,
        "summary": f"round {iter_idx + 1}: asked {asked}, added {added}",
    })

    return {
        "link_iter": iter_idx + 1,
        "link_changed": added > 0,
        "link_tried_pairs": list(tried),
        "stats": stats,
    }


def link_should_continue(state: PassState) -> str:
    return "link" if state.get("link_changed") else "finalize"
