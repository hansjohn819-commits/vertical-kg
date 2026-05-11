"""Persistent store for source-text chunks (§16.17).

Each entry is a single PDF page (after the §16.17.2 cutover to one-page
chunking, INGEST_PDF_PAGE_OVERLAP=0). Nodes and edges carry
`text_unit_ids: list[str]` pointing back to entries here, so the
chat composer can pull verbatim source text into evidence prompts
even though entity summaries were lossy LLM rewrites at extract time.

Persistence model mirrors GraphStorage: load all on boot to a
`dict[chunk_id, dict]`, save by atomic tmp→rename of one JSON file.
Volume estimate (5000 chunks × ~3 KB ≈ 15 MB) is comfortably in-memory;
swap for a sqlite/sharded backend only if data grows past ~100 MB.

Keys are flat uuid4 hex strings (no semantic prefix — see §16.17.6
decision 13). raw_doc_id and page_num are duplicated inside each
entry so reverse lookup ("which chunks belong to doc X?") is a
linear scan, which is fine at this scale.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable


class TextUnitStore:
    """In-memory dict + JSON file persistence."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._store: dict[str, dict] = {}

    # --- Mutations -------------------------------------------------------

    def add(self, unit: dict) -> None:
        """Add one chunk. `unit` must contain at least `id` (str). Idempotent
        on duplicate id — last write wins, which is fine because re-ingest
        of the same source produces identical text.
        """
        cid = unit["id"]
        self._store[cid] = unit

    def remove(self, chunk_id: str) -> bool:
        return self._store.pop(chunk_id, None) is not None

    # --- Queries ---------------------------------------------------------

    def get(self, chunk_id: str) -> dict | None:
        return self._store.get(chunk_id)

    def get_many(self, chunk_ids: Iterable[str]) -> list[dict]:
        """Return entries in the requested order, skipping missing ids
        silently. Caller decides whether missing ids are an error
        (boot consistency check is the canonical place for that)."""
        out: list[dict] = []
        for cid in chunk_ids:
            unit = self._store.get(cid)
            if unit is not None:
                out.append(unit)
        return out

    def keys(self) -> set[str]:
        return set(self._store.keys())

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._store

    def __len__(self) -> int:
        return len(self._store)

    def by_raw_doc(self, raw_doc_id: str) -> list[dict]:
        """Linear scan — fine at < 10K chunks. Sorted by page_num ascending
        so a reverse-source listing reads naturally."""
        hits = [u for u in self._store.values() if u.get("raw_doc_id") == raw_doc_id]
        hits.sort(key=lambda u: u.get("page_num", 0))
        return hits

    # --- Persistence -----------------------------------------------------

    def save(self) -> None:
        """Atomic write: dump to .tmp then rename. POSIX + Windows both
        provide atomic rename for same-filesystem moves, so a crash
        mid-write either leaves the old file intact or the new file
        complete — never half-written."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def load(self) -> bool:
        """Load existing file into memory. Returns True iff a file was
        loaded. Missing file → empty store (first boot / new instance).
        Corrupt file → raises (caller decides whether to wipe + rebuild
        — but text_units cannot be rebuilt from anywhere else, so a
        corrupt file is a real disaster, not auto-recoverable like
        the FAISS index in §16.10.5)."""
        if not self.path.exists():
            self._store = {}
            return False
        with self.path.open("r", encoding="utf-8") as f:
            self._store = json.load(f)
        return True
