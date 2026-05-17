"""4e: enforce ontology compliance on edge types (guide §16.19 c).

Lightweight, deterministic, zero-LLM. Runs before 4c. Sweeps every edge
through a five-branch decision tree:

    1. type ∈ alias_map               → rewrite to canonical
    5. type == RELATED_TO & original_type ∈ ontology & domain OK
                                      → type = original_type (keep original_type)
    2. type ∈ ontology & domain OK    → no-op
    3. type ∈ ontology & domain bad   → original_type = type; type = RELATED_TO
    4. type ∉ ontology & type ≠ RTO   → original_type = type; type = RELATED_TO

Each edge matches at most one branch per pass. Branch 5 is evaluated
before branch 2 because RELATED_TO is itself a registered relation, so
without the early check branch 2 would swallow rescue-eligible edges.
Alias resolution is final for the round; if the alias-resolved type
happens to violate domain, the next pass catches it via branch 3.
Worst case the graph converges in two passes per ontology revision.

Together with the schema field (`Edge.original_type`) added in §16.19 b,
this is what makes ontology evolution self-healing: once a previously
unregistered proposal gets added to the ontology, its existing
RELATED_TO edges flip back automatically the next time we wake up.
"""

from __future__ import annotations

from pathlib import Path

from src.graph.instance import GraphInstance
from src.graph.storage import GraphStorage

from .pass_log import log_event
from .state import PassState


def _domain_ok(
    rel_type: str, src_type: str, tgt_type: str,
    domain_map: dict[str, tuple[set[str], set[str]]],
) -> bool:
    """Local copy of m1_ingest._domain_ok to avoid importing M1 just for
    a 5-line helper. Unknown rel_type returns True — callers gate on
    `rel_type in domain_map` first."""
    if rel_type not in domain_map:
        return True
    src_set, tgt_set = domain_map[rel_type]
    if src_set and src_type not in src_set:
        return False
    if tgt_set and tgt_type not in tgt_set:
        return False
    return True


def _load_ontology_context(instance: GraphInstance):
    """Parse ontology + aliases — same shape link_form uses. Not cached:
    the user may edit ontology.md between passes and we want the new
    constraints visible immediately."""
    from src.modules.m1_ingest import parse_aliases, parse_ontology
    try:
        text = Path(instance.ontology_path).read_text(encoding="utf-8")
    except Exception:
        text = ""
    _ents, rels, dom = parse_ontology(text)
    aliases = parse_aliases(text)
    return rels, dom, aliases


def enforce_ontology_compliance(
    storage: GraphStorage,
    ontology_relation_types: set[str],
    domain_map: dict[str, tuple[set[str], set[str]]],
    alias_map: dict[str, str],
    *,
    pass_id: str | None = None,
) -> dict:
    """Pure decision-tree sweep over every edge. Returns counters.

    Caller is responsible for persisting storage afterward.
    """
    counters = {
        "examined": 0,
        "alias_rewrites": 0,
        "domain_downgrades": 0,
        "noise_downgrades": 0,
        "rescues": 0,
        "dangling_skipped": 0,
    }
    for e in list(storage.edges()):
        counters["examined"] += 1
        # Branch 1: alias rewrite. The M1 ingest path resolves aliases
        # at write time, so this is mostly a safety net for legacy
        # edges from before the alias mechanism existed, or for direct
        # edits to ontology aliases mid-life.
        if e.type in alias_map:
            canonical = alias_map[e.type]
            old = e.type
            e.type = canonical
            counters["alias_rewrites"] += 1
            log_event({
                "kind": "ontology_compliance",
                "subkind": "alias_rewrite",
                "pass_id": pass_id,
                "edge_id": e.id,
                "edge_type_before": old,
                "edge_type_after": canonical,
                "summary": f"alias rewrite {old} → {canonical}",
            })
            continue

        src = storage.get_node(e.source_id)
        tgt = storage.get_node(e.target_id)
        if src is None or tgt is None:
            counters["dangling_skipped"] += 1
            continue

        # Branch 5 (evaluated before 2 because RELATED_TO is in
        # ontology_relation_types itself; otherwise branch 2 would
        # swallow rescue-eligible edges).
        if (
            e.type == "RELATED_TO"
            and e.original_type
            and e.original_type in ontology_relation_types
            and _domain_ok(e.original_type, src.type, tgt.type, domain_map)
        ):
            counters["rescues"] += 1
            log_event({
                "kind": "ontology_compliance",
                "subkind": "rescue",
                "pass_id": pass_id,
                "edge_id": e.id,
                "edge_type_before": "RELATED_TO",
                "edge_type_after": e.original_type,
                "src_type": src.type, "src_label": src.label,
                "tgt_type": tgt.type, "tgt_label": tgt.label,
                "summary": (
                    f"rescue RELATED_TO → {e.original_type} "
                    f"(was {src.label} -> {tgt.label})"
                ),
            })
            e.type = e.original_type
            continue

        in_ontology = e.type in ontology_relation_types
        domain_passes = (
            _domain_ok(e.type, src.type, tgt.type, domain_map)
            if in_ontology
            else False
        )

        # Branch 2: registered & domain OK — leave alone.
        if in_ontology and domain_passes:
            continue

        # Branch 3: registered but used outside its domain — downgrade.
        if in_ontology and not domain_passes:
            counters["domain_downgrades"] += 1
            log_event({
                "kind": "ontology_compliance",
                "subkind": "domain_downgrade",
                "pass_id": pass_id,
                "edge_id": e.id,
                "edge_type_before": e.type,
                "edge_type_after": "RELATED_TO",
                "src_type": src.type, "src_label": src.label,
                "tgt_type": tgt.type, "tgt_label": tgt.label,
                "summary": (
                    f"domain downgrade {e.type} "
                    f"{src.type}→{tgt.type} (was {src.label} -> {tgt.label})"
                ),
            })
            e.original_type = e.type
            e.type = "RELATED_TO"
            continue

        # Branch 4: unregistered non-RELATED_TO type — noise sediment
        # left by Path 1 of `_classify_edge_type` in older runs.
        if not in_ontology and e.type != "RELATED_TO":
            counters["noise_downgrades"] += 1
            log_event({
                "kind": "ontology_compliance",
                "subkind": "noise_downgrade",
                "pass_id": pass_id,
                "edge_id": e.id,
                "edge_type_before": e.type,
                "edge_type_after": "RELATED_TO",
                "src_type": src.type, "src_label": src.label,
                "tgt_type": tgt.type, "tgt_label": tgt.label,
                "summary": (
                    f"noise downgrade {e.type} "
                    f"(was {src.label} -> {tgt.label})"
                ),
            })
            e.original_type = e.type
            e.type = "RELATED_TO"
            continue

        # Fall-through: legitimate RELATED_TO without a rescue-eligible
        # original_type, or RELATED_TO whose stash points to a type no
        # longer in ontology. Leave alone.

    return counters


def enforce_step(state: PassState, *, instance: GraphInstance) -> dict:
    pass_id = state.get("pass_id", "unknown")
    rels, dom, aliases = _load_ontology_context(instance)

    counters = enforce_ontology_compliance(
        instance.storage, rels, dom, aliases, pass_id=pass_id,
    )

    stats = dict(state.get("stats") or {})
    stats["ontology_compliance"] = counters

    log_event({
        "kind": "ontology_compliance_done",
        "pass_id": pass_id,
        "summary": (
            f"examined {counters['examined']}, "
            f"alias {counters['alias_rewrites']}, "
            f"domain_downgrades {counters['domain_downgrades']}, "
            f"noise_downgrades {counters['noise_downgrades']}, "
            f"rescues {counters['rescues']}, "
            f"dangling {counters['dangling_skipped']}"
        ),
        **counters,
    })

    return {"stats": stats}
