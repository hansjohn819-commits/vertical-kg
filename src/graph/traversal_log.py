"""Append-only JSONL log of what Q&A traversals touched, per guide §5.4.

4c reads this since the last pass, uses it to reinforce edge weights, then
clears it. Path sits next to the graph pickle so it rides along with the
instance.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from .storage import GraphStorage


def _log_path_for(storage: GraphStorage) -> Path:
    return storage.path.parent / "traversal.jsonl"


def append_query(
    storage: GraphStorage,
    question: str,
    seed_node_ids: list[str],
    touched_node_ids: list[str],
    touched_edge_ids: list[str],
) -> None:
    p = _log_path_for(storage)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": "graph_query",
        "question": question,
        "seed_node_ids": seed_node_ids,
        "touched_node_ids": touched_node_ids,
        "touched_edge_ids": touched_edge_ids,
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_all(storage: GraphStorage) -> list[dict]:
    p = _log_path_for(storage)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def clear(storage: GraphStorage) -> None:
    p = _log_path_for(storage)
    if p.exists():
        p.unlink()
