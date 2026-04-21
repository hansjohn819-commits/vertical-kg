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

    def neighbors(self, node_id: str) -> list[Node]:
        if node_id not in self._g:
            return []
        ids = set(self._g.successors(node_id)) | set(self._g.predecessors(node_id))
        return [n for n in (self.get_node(i) for i in ids) if n is not None]

    # --- Persistence ---

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as f:
            pickle.dump(self._g, f)

    def load(self) -> None:
        if self.path.exists():
            with self.path.open("rb") as f:
                self._g = pickle.load(f)

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
