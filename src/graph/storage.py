"""NetworkX-backed storage with pickle persistence.

Per guide §11, v1 is NetworkX. If data exceeds ~100k nodes, swap backends
behind this class's interface — callers should not depend on nx details.
"""

import pickle
from collections.abc import Iterator
from pathlib import Path

import networkx as nx

from .models import Edge, Node


class GraphStorage:
    def __init__(self, storage_path: str | Path):
        self.path = Path(storage_path)
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()

    # --- Nodes ---

    def add_node(self, node: Node) -> None:
        self._g.add_node(node.id, data=node)

    def get_node(self, node_id: str) -> Node | None:
        if node_id not in self._g:
            return None
        return self._g.nodes[node_id].get("data")

    def remove_node(self, node_id: str) -> None:
        if node_id in self._g:
            self._g.remove_node(node_id)

    def nodes(self) -> Iterator[Node]:
        for _, attrs in self._g.nodes(data=True):
            data = attrs.get("data")
            if data is not None:
                yield data

    # --- Edges ---

    def add_edge(self, edge: Edge) -> int:
        """Returns the nx multi-edge key so callers can address this specific edge."""
        return self._g.add_edge(edge.source_id, edge.target_id, data=edge)

    def edges(self) -> Iterator[Edge]:
        for _, _, attrs in self._g.edges(data=True):
            data = attrs.get("data")
            if data is not None:
                yield data

    def get_edge(self, edge_id: str) -> Edge | None:
        for e in self.edges():
            if e.id == edge_id:
                return e
        return None

    def remove_edge_by_id(self, edge_id: str) -> bool:
        for u, v, k, attrs in list(self._g.edges(keys=True, data=True)):
            data = attrs.get("data")
            if data is not None and data.id == edge_id:
                self._g.remove_edge(u, v, key=k)
                return True
        return False

    def incident_edges(self, node_id: str) -> list[Edge]:
        if node_id not in self._g:
            return []
        out: list[Edge] = []
        for u, v, attrs in self._g.out_edges(node_id, data=True):
            data = attrs.get("data")
            if data is not None:
                out.append(data)
        for u, v, attrs in self._g.in_edges(node_id, data=True):
            data = attrs.get("data")
            if data is not None:
                out.append(data)
        return out

    def neighbors(self, node_id: str) -> list[Node]:
        if node_id not in self._g:
            return []
        ids = set(self._g.successors(node_id)) | set(self._g.predecessors(node_id))
        return [n for n in (self.get_node(i) for i in ids) if n is not None]

    def degree(self, node_id: str) -> int:
        """Total degree (in + out). Used by link_form bridge detection."""
        if node_id not in self._g:
            return 0
        return self._g.in_degree(node_id) + self._g.out_degree(node_id)

    def edge_types_between(self, u: str, v: str) -> list[str]:
        """Edge types connecting `u` and `v` in either direction.

        MultiDiGraph can have multiple edges between the same pair —
        return all of their types. Used by link_form to inspect the
        edges of a BFS path for the containment-pattern check.
        """
        types: list[str] = []
        for src, dst in ((u, v), (v, u)):
            edge_data = self._g.get_edge_data(src, dst)
            if edge_data is None:
                continue
            for _key, attrs in edge_data.items():
                data = attrs.get("data")
                if data is not None:
                    types.append(data.type)
        return types

    # --- Edge dedup (§16.17 problem 2) ---

    def dedupe_edges(self, predicate=None) -> dict:
        """Merge duplicate edges sharing the same (source_id, target_id, type).

        Real-world trigger: in a multi-page PDF, page headers / author
        footers / repeated tables cause M1 to extract the *same*
        relationship N times (once per page that mentions both
        endpoints). Without dedup the graph carries N copies that all
        say the same thing. This is the cleanup step.

        Per group: pick the lowest-id edge as the survivor (deterministic),
        merge the rest into it:
          * text_unit_ids = sorted union of every edge's chunk pointers
          * weight = sum of all weights (so "5 pages all confirm this"
            shows up as a heavier edge — useful signal for §16.17 §16.10)
          * evidence_quote = the longest non-empty quote across the group
            (a longer quote is usually a fuller / more informative span)
          * retraction_log = concatenation of all retraction events,
            preserving the audit trail.

        ``predicate(edge) -> bool`` filters which edges are eligible.
        Default is "every edge in the graph" — used by the one-off
        cleanup script. M1's ingest-end call passes a per-run filter so
        only edges from the *current* ingest are considered (cross-doc
        duplicates carry independent confirmation and stay distinct).

        Returns ``{"groups_merged": int, "edges_removed": int}``.
        """
        from collections import defaultdict

        groups: dict[tuple[str, str, str], list[Edge]] = defaultdict(list)
        for e in self.edges():
            if predicate is None or predicate(e):
                groups[(e.source_id, e.target_id, e.type)].append(e)

        groups_merged = 0
        edges_removed = 0
        for group in groups.values():
            if len(group) <= 1:
                continue
            group.sort(key=lambda e: e.id)
            winner = group[0]
            chunk_set: set[str] = set(winner.text_unit_ids or [])
            total_weight = winner.weight
            best_quote = winner.evidence_quote or ""
            for other in group[1:]:
                chunk_set.update(other.text_unit_ids or [])
                total_weight += other.weight
                if other.evidence_quote and len(other.evidence_quote) > len(best_quote):
                    best_quote = other.evidence_quote
                if other.retraction_log:
                    winner.retraction_log = list(winner.retraction_log) + list(other.retraction_log)
                self.remove_edge_by_id(other.id)
                edges_removed += 1
            winner.text_unit_ids = sorted(chunk_set)
            winner.weight = total_weight
            winner.evidence_quote = best_quote
            groups_merged += 1

        return {"groups_merged": groups_merged, "edges_removed": edges_removed}

    # --- Persistence ---

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as f:
            pickle.dump(self._g, f)

    def load(self) -> None:
        if self.path.exists():
            with self.path.open("rb") as f:
                self._g = pickle.load(f)
            self._backfill_new_fields()

    def _backfill_new_fields(self) -> None:
        """Re-validate every Node/Edge through pydantic so fields added to
        the model after the pickle was written get their default values.

        Pickled pydantic instances retain only the fields present at save
        time; accessing a newly-added field on an old instance raises
        AttributeError. This pass dumps each model to dict and re-builds
        it, which fills missing fields with the model's declared defaults.
        Runs once at load. Cost is negligible (each node/edge is a tiny
        dict round-trip; 142 nodes ≈ <10ms).

        Used so the §16.17 schema additions (`Node.text_unit_ids`,
        `Edge.text_unit_ids`, `Edge.evidence_quote`) become accessible on
        graphs that were saved before the schema change. The fields
        default to empty list / empty string so the consistency check
        treats old data as "no source-text linkage yet" — exactly what
        we want until re-ingest.
        """
        for nid, attrs in list(self._g.nodes(data=True)):
            n = attrs.get("data")
            if n is None:
                continue
            try:
                n.text_unit_ids  # noqa: B018 — probe attr presence
                continue
            except AttributeError:
                pass
            attrs["data"] = Node(**n.model_dump())
        for u, v, k, attrs in list(self._g.edges(keys=True, data=True)):
            e = attrs.get("data")
            if e is None:
                continue
            try:
                e.text_unit_ids  # noqa: B018
                e.evidence_quote  # noqa: B018
                continue
            except AttributeError:
                pass
            attrs["data"] = Edge(**e.model_dump())

    # --- Stats ---

    def stats(self) -> dict:
        node_types: dict[str, int] = {}
        for n in self.nodes():
            node_types[n.type] = node_types.get(n.type, 0) + 1
        edge_types: dict[str, int] = {}
        for e in self.edges():
            edge_types[e.type] = edge_types.get(e.type, 0) + 1
        return {
            "node_count": self._g.number_of_nodes(),
            "edge_count": self._g.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
        }
