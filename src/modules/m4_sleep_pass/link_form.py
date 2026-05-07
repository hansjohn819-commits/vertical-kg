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
from .state import (
    LINK_BFS_DEPTH,
    LINK_HUB_DEGREE_THRESHOLD,
    LINK_MAX_NEW_PER_ROUND,
    PassState,
)

LINK_SYSTEM_PROMPT = """You are a knowledge-graph discovery judge. You look
at two nodes (plus the path of intermediate nodes that link them) and decide
whether a DIRECT semantic relation exists between the two endpoints — one
that the SEED node's or REMOTE node's own summary explicitly supports.

Reply with STRICT JSON only, exactly these keys:
{
  "holds": true | false,
  "direction": "seed_to_remote" | "remote_to_seed",
  "what": "<name of the relation, or empty string if holds=false>",
  "why": "<one or two sentences. MUST contain a verbatim quote (in single
          quotes) from the seed's or remote's summary that supports the
          relation. If neither summary contains support, holds=false.>",
  "edge_type": "<specific UPPER_SNAKE_CASE verb describing the relation>",
  "evidence_from": "seed" | "remote" | "both"
}

CRITICAL — set holds=false when ANY of these apply:

1. CATEGORY CO-MEMBERSHIP. If the only thing connecting A and B is that
   they're both in industry X / country X / species family X, that's
   category co-membership, not a relation. Reject.

2. TRANSITIVE CLOSURE. "Sugar kelp grows in Maine, Maine is in US, therefore
   Sugar kelp FARMED_IN US" — the chain is already in the graph, a direct
   edge adds nothing. Reject.

3. WORLD-KNOWLEDGE INFERENCE. "China and Norway are both major seafood
   exporters, therefore MARKET_COMPETITOR" — true in the world but not
   from these nodes' summaries. Reject.

4. EVIDENCE COMES FROM AN INTERMEDIATE NODE, NOT THE ENDPOINTS. Path
   A→C→B where C's summary says "C does X to A" and "C does Y to B" — that
   describes two independent C-relations, NOT a relation between A and B.
   The verbatim quote in `why` MUST come from A's or B's own summary, not
   from C. Reject if you can only support the relation by citing C.

5. VAGUE / GENERIC RELATION. The `edge_type` must be a specific verb that
   describes what the relation actually does. FORBIDDEN edge_types:
   RELATED_TO, ASSOCIATED_WITH, CONNECTED_TO, LINKED_TO, INVOLVES, ABOUT,
   CONCERNS, RELATES_TO, PERTAINS_TO. If you want to use one of these, the
   relation isn't specific enough — reject.

6. THE NATURAL DIRECTION IS NEITHER seed→remote NOR remote→seed. If the
   verb's natural subject is the intermediate node, not either endpoint,
   the relation isn't between the endpoints. Reject.

DIRECTION RULES:
  - "seed_to_remote": the verb's subject is the seed node (e.g., FAO
    MANAGES_STATISTICS_FOR Capture Fisheries → seed=FAO, direction=seed_to_remote)
  - "remote_to_seed": the verb's subject is the remote node (e.g., remote
    FUNDS seed → direction=remote_to_seed)
  - If neither direction sounds natural, set holds=false.

When holds=true, the `why` MUST contain a verbatim quoted phrase (≤30
words, in single quotes) lifted directly from the seed's or remote's
summary. The phrase should be the textual evidence for the relation.
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


# Containment-style edge-type patterns (universal English semantic
# patterns, not domain-specific). M1 routinely emits verbs like
# `is_part_of`, `located_in`, `is_subsector_of`, `contains`, `includes`,
# `belongs_to`, etc. Two containment-style edges in a row = pure
# transitive closure (A is in C, C is in/contains B → "A relates to B"
# is co-membership, not a real relation). Closed semantic category
# means we don't need to extend this list when M1 invents new
# domain-specific verbs in future corpora.
_CONTAINMENT_EDGE_PATTERNS = [
    # "is X" / "is X of" / "is_X_of" — covers is_a, is_part_of,
    # is_subsector_of, is_member_of, is_source_of, is_kind_of, is_type_of.
    re.compile(r"^IS[-_ ](A|AN|THE|PART|MEMBER|TYPE|KIND|INSTANCE|SUBSECTOR|SOURCE)([-_ ]OF)?$", re.I),
    re.compile(r"^IS[-_ ]A[-_ ](TYPE|KIND|MEMBER|PART|SUBSECTOR|FORM)([-_ ]OF)?$", re.I),  # "is a type of"
    # X_OF / X of — part_of, member_of, kind_of, type_of, etc.
    re.compile(r"^(PART|MEMBER|TYPE|KIND|SUBSET|INSTANCE|CATEGORY|SUBSECTOR)[-_ ]OF$", re.I),
    # *_to forms — belongs_to, refers_to, relates_to, pertains_to.
    re.compile(r"^(BELONGS|REFERS|RELATES|PERTAINS)[-_ ]TO$", re.I),
    # Anything ending in _in / at / under / inside / within / from
    # — covers found_in, located_in, operates_in, based_in, set_in, etc.
    re.compile(r".+[-_ ](IN|AT|UNDER|INSIDE|WITHIN|FROM)$", re.I),
    # Bare prepositions (M1 sometimes emits these as edge types).
    re.compile(r"^(IN|AT|FROM|UNDER|INSIDE|WITHIN|OF)$", re.I),
    # Containment verbs in the other direction (target contains source):
    # contains, includes, comprises, encompasses, hosts, has_member, has_part.
    re.compile(r"^(CONTAINS?|INCLUDES?|COMPRISES?|ENCOMPASSES?|HOSTS?|HOLDS|HAS[-_ ](MEMBER|TYPE|PART|INSTANCES?))$", re.I),
]


def _is_containment_edge(edge_type: str) -> bool:
    et = (edge_type or "").strip()
    if not et:
        return False
    return any(p.match(et) for p in _CONTAINMENT_EDGE_PATTERNS)


def _path_is_bridge(storage: GraphStorage, path: list[str]) -> bool:
    """Pre-LLM topology filter (§16 link_form 二次收紧 2026-05-06).

    Two complementary signals; either fires → path is a bridge → skip:

    Signal 1 — CONTAINMENT EDGE PATTERN (semantic, cold-start safe):
      Every edge along `path` matches a containment-style verb pattern
      (is_part_of, located_in, contains, includes, ...). This catches
      pure transitive-closure shortcuts regardless of graph maturity.
      Works at cold-start because edge types are M1-determined.

    Signal 2 — HUB DEGREE THRESHOLD (structural, post-merge):
      Every intermediate node has degree >= LINK_HUB_DEGREE_THRESHOLD.
      Catches institutional hub bridges (FAO routes BT→Qu Dongyu) that
      use real action verbs but route through a hub. Activates once the
      graph matures via merge.

    Direct neighbors (path length ≤ 2) have no intermediates → never
    a bridge.
    """
    if len(path) <= 2:
        return False

    # Signal 1: walk the path, every consecutive pair must have at least
    # one containment-style edge between them.
    edges_all_containment = True
    for i in range(len(path) - 1):
        edge_types = storage.edge_types_between(path[i], path[i + 1])
        if not any(_is_containment_edge(et) for et in edge_types):
            edges_all_containment = False
            break
    if edges_all_containment:
        return True

    # Signal 2: every intermediate is a hub by total-degree.
    intermediates = path[1:-1]
    if intermediates and all(
        storage.degree(pid) >= LINK_HUB_DEGREE_THRESHOLD for pid in intermediates
    ):
        return True

    return False


# Vague edge_types that the prompt forbids — code-level enforcement so the
# rule doesn't depend on Gemma reading instructions perfectly. Match is
# case-insensitive on the normalized UPPER_SNAKE_CASE.
_FORBIDDEN_VAGUE_EDGE_TYPES = frozenset({
    "RELATED_TO", "ASSOCIATED_WITH", "CONNECTED_TO", "LINKED_TO",
    "INVOLVES", "ABOUT", "CONCERNS", "RELATES_TO", "PERTAINS_TO",
})

# Single-quoted phrase capture: the prompt asks the LLM to wrap a verbatim
# quote in single quotes. We accept ASCII apostrophes plus the curly
# variants that PDFs / smart-quote keyboards routinely emit.
_QUOTE_RE = re.compile(r"['‘’]([^'‘’]{4,})['‘’]")


def _normalize_for_substring(s: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation we don't want to
    block on. Used for the verbatim-quote substring check — we want
    'manages global food and fisheries statistics' to match even if the
    summary has a comma the LLM dropped.
    """
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[,\.;:!?\"]", "", s)
    return s.strip()


def _why_has_grounded_quote(why: str, seed_summary: str, remote_summary: str) -> bool:
    """True iff `why` contains at least one single-quoted phrase that
    actually appears (loosely) in seed_summary or remote_summary.

    This is the code-level guard against the LLM fabricating quotes —
    prompt asks for verbatim, but Gemma sometimes paraphrases. We
    substring-match the lowercased / whitespace-normalized form so trivial
    punctuation drift doesn't fail a real quote.
    """
    quotes = _QUOTE_RE.findall(why)
    if not quotes:
        return False
    haystack = _normalize_for_substring(seed_summary + " " + remote_summary)
    for q in quotes:
        needle = _normalize_for_substring(q)
        if needle and needle in haystack:
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
    skipped_topology = 0
    rejected_quote = 0
    rejected_vague = 0
    rejected_direction = 0
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
            # Pre-LLM topology filter: multi-signal bridge detection
            # (containment edge pattern + hub degree). See _path_is_bridge.
            if _path_is_bridge(storage, path):
                skipped_topology += 1
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
                # thinking=False: link judge with thinking-on tends to
                # rationalize trivial relations into plausible-sounding
                # edges (observed 2026-05-06 — China→CONSUMES→Fishmeal,
                # Sugar kelp→FARMED_IN→US 等传递闭包)。关 thinking 让模型
                # 守住 prompt 的"corpus-grounded only"硬规则，且 ~15× 更快。
                resp = client.chat(
                    messages=[
                        {"role": "system", "content": LINK_SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.1,
                    thinking=False,
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

            edge_type = str(parsed.get("edge_type", "")).strip().upper()
            if not edge_type:
                edge_type = "RELATED_TO"

            # Code-level enforcement of the prompt's forbidden-vague rule.
            # Prompt says no RELATED_TO/etc.; this guard catches model
            # backsliding and keeps the rule deterministic.
            if edge_type in _FORBIDDEN_VAGUE_EDGE_TYPES:
                rejected_vague += 1
                log_event({
                    "kind": "link_rejected_vague_type",
                    "pass_id": pass_id,
                    "summary": f"{seed_node.label} ~ {remote_node.label}: vague type {edge_type}",
                    "edge_type": edge_type,
                    "what": what,
                })
                continue

            # Direction handling: LLM tells us which way the relation flows.
            # Default seed_to_remote when missing for backward compat. The
            # prompt also asks the LLM to set holds=false when neither
            # direction is natural — this code path just respects the call.
            direction = str(parsed.get("direction", "seed_to_remote")).strip().lower()
            if direction not in ("seed_to_remote", "remote_to_seed"):
                rejected_direction += 1
                log_event({
                    "kind": "link_rejected_bad_direction",
                    "pass_id": pass_id,
                    "summary": f"{seed_node.label} ~ {remote_node.label}: direction={direction!r}",
                    "direction": direction,
                })
                continue
            if direction == "seed_to_remote":
                src_id, tgt_id = seed_id, remote_id
                src_node, tgt_node = seed_node, remote_node
            else:
                src_id, tgt_id = remote_id, seed_id
                src_node, tgt_node = remote_node, seed_node

            # Verbatim-quote grounding check. The prompt asks for a single-
            # quoted phrase from seed/remote summary inside `why`. We
            # substring-match (loosely) to detect fabricated quotes. If the
            # LLM's evidence comes from the path's intermediate node, no
            # endpoint-summary substring will match → reject. This is the
            # mechanical guard against the "Peter King → MANAGES_PRIORITY"
            # / "Blue Transformation → PUBLISHED_BY → SOFIA" failure modes.
            if not _why_has_grounded_quote(why, seed_node.summary, remote_node.summary):
                rejected_quote += 1
                log_event({
                    "kind": "link_rejected_no_quote",
                    "pass_id": pass_id,
                    "summary": f"{seed_node.label} ~ {remote_node.label}: why lacks endpoint-summary quote",
                    "why": why[:200],
                })
                continue

            new_edge = Edge(
                source_id=src_id,
                target_id=tgt_id,
                type=edge_type,
                weight=0.5,  # new derived edges start mid-weight (§5.4 PageRank-like)
                provenance=DerivedProv(
                    operation_id=f"{pass_id}:link:{src_node.label}->{tgt_node.label}",
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
                "summary": f"{src_node.label} -[{edge_type}]-> {tgt_node.label}",
                "edge_id": new_edge.id,
                "edge_type": edge_type,
                "source_id": src_id,
                "source_label": src_node.label,
                "target_id": tgt_id,
                "target_label": tgt_node.label,
                "direction": direction,
                "what": what,
                "why": why,
                "basis": path_summary,
            })
        if added >= LINK_MAX_NEW_PER_ROUND:
            break

    stats = dict(state.get("stats") or {})
    stats["link_total"] = int(stats.get("link_total", 0)) + added
    stats[f"link_round_{iter_idx}"] = {
        "asked": asked,
        "added": added,
        "skipped_topology": skipped_topology,
        "rejected_quote": rejected_quote,
        "rejected_vague": rejected_vague,
        "rejected_direction": rejected_direction,
    }

    log_event({
        "kind": "link_round_done",
        "pass_id": pass_id,
        "summary": (
            f"round {iter_idx + 1}: asked {asked}, added {added}, "
            f"skipped_topology {skipped_topology}, "
            f"rejected_quote {rejected_quote}, "
            f"rejected_vague {rejected_vague}, "
            f"rejected_direction {rejected_direction}"
        ),
    })

    return {
        "link_iter": iter_idx + 1,
        "link_changed": added > 0,
        "link_tried_pairs": list(tried),
        "stats": stats,
    }


def link_should_continue(state: PassState) -> str:
    return "link" if state.get("link_changed") else "finalize"
