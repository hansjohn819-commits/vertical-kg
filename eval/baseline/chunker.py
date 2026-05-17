"""Independent chunker for the RAG baseline.

Conventional RAG chunking — not the per-page entity-extraction chunking the
main project uses. Token-based with overlap so passages are long enough to
carry context and overlap-bridged so a sentence near a chunk boundary still
appears whole somewhere in the index.

Config:
    CHUNK_TOKENS   = 500     # target chunk size
    OVERLAP_TOKENS = 100     # 20% overlap

Per-PDF flow (handled by build_baseline_index.py):
    1. extract per-page text via pdfplumber
    2. concat with page markers like "\\n\\n[PAGE 17]\\n\\n"
    3. recursive split on sep priority ['\\n\\n', '\\n', '. ', ' ', '']
       respecting token budget
    4. record per-chunk metadata: {id, doc_path, doc_title, page_start,
       page_end, text, n_tokens}
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

import tiktoken

CHUNK_TOKENS = 500
OVERLAP_TOKENS = 100
ENCODER = tiktoken.get_encoding("cl100k_base")

_PAGE_MARKER_RE = re.compile(r"\[PAGE (\d+)\]")


@dataclass
class Chunk:
    id: str
    doc_path: str
    doc_title: str
    page_start: int
    page_end: int
    text: str
    n_tokens: int


def _ntokens(s: str) -> int:
    return len(ENCODER.encode(s))


def _split_by_separators(text: str, max_tokens: int) -> list[str]:
    """Recursive char splitter — try big separators first, fall back to char."""
    seps = ["\n\n", "\n", ". ", " ", ""]

    def rec(t: str, sep_idx: int) -> list[str]:
        if _ntokens(t) <= max_tokens:
            return [t]
        if sep_idx >= len(seps):
            # Hard char chop.
            tokens = ENCODER.encode(t)
            return [ENCODER.decode(tokens[i:i + max_tokens])
                    for i in range(0, len(tokens), max_tokens)]
        sep = seps[sep_idx]
        if sep == "":
            return rec(t, sep_idx + 1)
        parts = t.split(sep)
        if len(parts) == 1:
            return rec(t, sep_idx + 1)
        # Re-attach separator (except last part) so re-joining preserves text.
        rejoined: list[str] = []
        for i, p in enumerate(parts):
            if i < len(parts) - 1:
                rejoined.append(p + sep)
            else:
                rejoined.append(p)
        out: list[str] = []
        for p in rejoined:
            if _ntokens(p) <= max_tokens:
                out.append(p)
            else:
                out.extend(rec(p, sep_idx + 1))
        return out

    return rec(text, 0)


def _merge_with_overlap(small_pieces: list[str]) -> list[str]:
    """Greedy merge small pieces into chunks of ~CHUNK_TOKENS, then add
    OVERLAP_TOKENS overlap between adjacent chunks.

    Overlap is applied by appending the *tail* of the previous chunk to the
    *head* of the next so retrieval can still hit a passage that straddled
    the boundary.
    """
    target = CHUNK_TOKENS
    chunks: list[str] = []
    cur = ""
    cur_tok = 0
    for p in small_pieces:
        pt = _ntokens(p)
        if cur and cur_tok + pt > target:
            chunks.append(cur.strip())
            cur = p
            cur_tok = pt
        else:
            cur = (cur + p) if cur else p
            cur_tok += pt
    if cur.strip():
        chunks.append(cur.strip())

    if len(chunks) <= 1 or OVERLAP_TOKENS <= 0:
        return chunks

    # Apply overlap by prepending the last OVERLAP_TOKENS of chunk[i-1] to chunk[i].
    out: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tokens = ENCODER.encode(chunks[i - 1])
        tail = ENCODER.decode(prev_tokens[-OVERLAP_TOKENS:])
        out.append(tail + " " + chunks[i])
    return out


def _extract_pages_for_chunk(text: str) -> tuple[int, int]:
    """Find first/last [PAGE N] marker in the chunk text."""
    matches = _PAGE_MARKER_RE.findall(text)
    if not matches:
        return (-1, -1)
    nums = [int(m) for m in matches]
    return (min(nums), max(nums))


def chunk_document(
    doc_path: str,
    doc_title: str,
    pages_text: list[str],
) -> list[Chunk]:
    """Split a paginated document into baseline chunks.

    pages_text: list of page strings (1-indexed by position+1).
    Returns list[Chunk]; each chunk's text contains [PAGE N] markers so we
    can recover its page range for gold-page recall scoring.
    """
    # Build a single doc string with page markers.
    parts: list[str] = []
    for i, ptext in enumerate(pages_text):
        page_num = i + 1
        body = (ptext or "").strip()
        if not body:
            continue
        parts.append(f"\n\n[PAGE {page_num}]\n\n{body}")
    full = "".join(parts).strip()
    if not full:
        return []

    small = _split_by_separators(full, CHUNK_TOKENS)
    merged = _merge_with_overlap(small)

    out: list[Chunk] = []
    for text in merged:
        ps, pe = _extract_pages_for_chunk(text)
        out.append(Chunk(
            id=uuid.uuid4().hex,
            doc_path=doc_path,
            doc_title=doc_title,
            page_start=ps,
            page_end=pe,
            text=text,
            n_tokens=_ntokens(text),
        ))
    return out
