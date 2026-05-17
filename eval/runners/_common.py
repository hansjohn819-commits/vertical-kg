"""Shared utilities for SUT runners.

- load_gold(): read eval/testset/gold.jsonl
- write_run_record(): append a JSONL record to raw_runs/<system>.jsonl
- get_selected_chunks_internal(): replicate _build_chunk_evidence selection
  to capture which project chunk_ids make it into the prompt for internal
  / external paths. Necessary because the actual project function only
  returns (text, included_node_ids); we need chunk_ids for unified scoring.
- chunk_to_evidence_pages(): given a project chunk_id, return (raw_doc_id,
  page_num) for unified page-level recall scoring.
- baseline_chunk_to_evidence_pages(): given a baseline chunk dict, return
  (raw_doc_id, page_num) tuples (one chunk may span 2 pages).
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GOLD_PATH = ROOT / "eval" / "testset" / "gold.jsonl"
RAW_RUNS_DIR = ROOT / "eval" / "reports" / "raw_runs"


def load_gold() -> list[dict]:
    out = []
    with GOLD_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def open_run_writer(system_name: str):
    RAW_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_RUNS_DIR / f"{system_name}.jsonl"
    return path.open("w", encoding="utf-8"), path


def write_run_record(fh, rec: dict) -> None:
    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    fh.flush()


@contextmanager
def timed():
    t0 = time.time()
    holder = {"ms": None}
    yield holder
    holder["ms"] = int((time.time() - t0) * 1000)


# -------- Chunk selection mirror for internal / external paths --------

def get_selected_chunks(instance, question: str,
                        use_display_title: bool) -> tuple[list[str], list[dict]]:
    """Default seed path: top_k(k=10) + chunk selection. Convenience wrapper
    around get_selected_chunks_for_seeds() for the v1 runners.
    """
    from src.graph.retrieval import top_k
    seeds = top_k(instance.vector_store, instance.storage, question, k=10)
    return get_selected_chunks_for_seeds(
        instance, question, seeds, use_display_title=use_display_title,
    )


def get_selected_chunks_for_seeds(
    instance, question: str, seeds: list,
    *, use_display_title: bool,
) -> tuple[list[str], list[dict]]:
    """Replicate the selection inside src.modules.m2_qa_agent._build_chunk_evidence,
    but accept pre-computed seeds (so external_v2 can swap in hybrid retrieval).

    Returns (selected_chunk_ids_in_rank_order, seed_node_dicts).

    Why a copy and not a wrapper: the project function returns text + node_ids,
    not chunk_ids. We need chunk_ids for unified retrieval recall scoring.
    """
    from src.graph.retrieval import encode_query, _get_model
    from src.graph.tokens import count_tokens
    from src.modules.m2_qa_agent import (
        _RERANK_TOP_N_CHUNKS, RETRIEVAL_BUDGET_TOKENS,
    )

    text_units = instance.text_units

    seed_chunk_ids: dict[str, list[str]] = {}
    chunk_payloads: dict[str, dict] = {}
    for seed in seeds:
        ids: list[str] = []
        for cid in (getattr(seed, "text_unit_ids", None) or []):
            cd = text_units.get(cid)
            if cd is None:
                continue
            ids.append(cid)
            chunk_payloads.setdefault(cid, cd)
        seed_chunk_ids[seed.id] = ids

    seeds_payload = [{
        "id": s.id, "label": s.label, "type": s.type,
    } for s in seeds]

    if not chunk_payloads:
        return [], seeds_payload

    qv = encode_query(question)
    model = _get_model()
    all_cids = list(chunk_payloads)
    chunk_texts = [(chunk_payloads[c].get("text", "") or "") for c in all_cids]
    chunk_vecs = model.encode(
        chunk_texts, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores = (chunk_vecs @ qv).tolist()
    ranked = sorted(zip(all_cids, scores), key=lambda x: -x[1])
    chunk_rank = {cid: i for i, (cid, _) in enumerate(ranked)}

    selected: list[str] = []
    used = 0
    for cid, _ in sorted(chunk_rank.items(), key=lambda x: x[1]):
        if len(selected) >= _RERANK_TOP_N_CHUNKS:
            break
        cd = chunk_payloads[cid]
        rid = cd.get("raw_doc_id", "") or ""
        page = cd.get("page_num")
        if use_display_title:
            from src.graph.retrieval import display_title
            title = display_title(rid)
        else:
            title = rid
        header = (
            f'  [from "{title}", page {page}]'
            if page is not None else f'  [from "{title}"]'
        )
        body = "  " + (cd.get("text", "") or "").strip()
        rendered = header + "\n" + body
        ct = count_tokens(rendered)
        if used + ct > RETRIEVAL_BUDGET_TOKENS:
            continue
        selected.append(cid)
        used += ct

    return selected, seeds_payload


def chunk_to_evidence_page(instance, chunk_id: str) -> dict | None:
    cd = instance.text_units.get(chunk_id)
    if cd is None:
        return None
    return {"raw_doc_id": cd.get("raw_doc_id", ""), "page_num": cd.get("page_num")}


def doc_basename_from_baseline_path(doc_path: str) -> str:
    return Path(doc_path).name


def baseline_chunk_to_evidence_pages(chunk: dict) -> list[dict]:
    """A baseline chunk may span pages [page_start, page_end]. Return all
    (raw_doc_id, page_num) tuples it covers.
    raw_doc_id is the basename of the source PDF, matching how the project
    stores raw_doc_id (see src/modules/m2_qa_agent.py _impl_ingest_file).
    """
    raw_doc_id = doc_basename_from_baseline_path(chunk.get("doc_path", ""))
    ps = chunk.get("page_start")
    pe = chunk.get("page_end")
    if ps is None or pe is None or ps < 0 or pe < 0:
        return [{"raw_doc_id": raw_doc_id, "page_num": None}]
    return [{"raw_doc_id": raw_doc_id, "page_num": p} for p in range(ps, pe + 1)]


# -------- Cold/warm-up handling --------

def warmup(instance=None) -> None:
    """Trigger one-time imports + model loads so subsequent calls are warm.

    For internal/external: runs sentence-transformers _get_model() and one
    cheap query through the LocalClient (a tiny no-op chat).

    For baseline: callers do their own warm-up since they have an independent
    embedding model load.
    """
    from src.graph.retrieval import _get_model
    _get_model()
    if instance is not None:
        # Touch storage / vector store to ensure FAISS is loaded.
        instance.storage.stats()
