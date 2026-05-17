"""Baseline RAG composer.

Standard "stuff retrieved chunks into prompt + ask LLM" pattern. Uses the
same LocalClient as the main project (so LLM capability is constant across
SUTs — the comparison is purely about retrieval / system architecture, not
about model strength).

Token budget for evidence: 6000 tokens. Conventional RAG implementations
use 4-8k of context for top-k chunks; we leave plenty of room for the
question + answer in a 40k effective window.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.graph.tokens import count_tokens
from src.llm.local_client import LocalClient

EVIDENCE_BUDGET_TOKENS = 6000

BASELINE_SYSTEM_PROMPT = (
    "You are answering a user question using the document excerpts provided "
    "below. The excerpts come from a corpus of industry reports. Each excerpt "
    "is labeled with its source document title.\n\n"
    "Hard rules:\n"
    "1. Answer ONLY from the excerpts. Do not add facts from prior training.\n"
    "2. When you cite a fact, mention the source document by title in the "
    "sentence (e.g., \"according to the Maine Seaweed Benchmarking Report\").\n"
    "3. If the excerpts do not contain enough information to answer the "
    "question, say plainly that you don't have enough information. Do not "
    "speculate, and do not fall back on general knowledge.\n"
    "4. Be concise — write a direct answer in natural prose, not a summary "
    "of the excerpts."
)


def _build_evidence_block(chunks: list[dict],
                          budget_tokens: int = EVIDENCE_BUDGET_TOKENS) -> str:
    blocks = []
    used = 0
    for i, c in enumerate(chunks, start=1):
        title = c.get("doc_title", "")
        ps = c.get("page_start", -1)
        pe = c.get("page_end", -1)
        if ps == pe and ps >= 0:
            page_label = f"page {ps}"
        elif ps >= 0 and pe >= 0:
            page_label = f"pages {ps}-{pe}"
        else:
            page_label = ""
        head = f"[Excerpt {i} — \"{title}\"" + (f", {page_label}" if page_label else "") + "]"
        body = (c.get("text", "") or "").strip()
        block = head + "\n" + body
        bt = count_tokens(block)
        if used + bt > budget_tokens:
            break
        blocks.append(block)
        used += bt
    return "\n\n".join(blocks)


def compose(question: str, chunks: list[dict],
            client: LocalClient | None = None) -> str:
    """Run the baseline composer LLM call. Returns answer text.

    `chunks` is the list of retrieved chunk dicts (each with text + doc_title
    + page_start/end). They get formatted into the EVIDENCE block in order.
    """
    client = client or LocalClient()
    evidence = _build_evidence_block(chunks)
    if not evidence:
        evidence = "(no excerpts found)"
    user = f"EXCERPTS:\n{evidence}\n\nQUESTION: {question}\n\nANSWER:"
    resp = client.chat(
        messages=[
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        thinking=False,  # match external for fair latency comparison + speed
    )
    return resp.choices[0].message.content or ""
