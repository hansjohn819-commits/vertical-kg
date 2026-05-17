"""Build the baseline RAG index from data/raw/*.pdf.

Independent from the main project's chunking / vector store. Produces
files in eval/data/baseline_index/:
  - chunks.pkl             list[dict] with text + doc_title + page_start/end
  - dense.faiss            FAISS IP index over chunk embeddings
  - dense_ids.json         row -> chunk_id mapping
  - bm25.pkl               rank_bm25 BM25Okapi over chunk tokens
  - manifest.json          run metadata
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pdfplumber  # noqa: E402

from eval.baseline.bm25_store import BaselineBM25Store  # noqa: E402
from eval.baseline.chunker import chunk_document  # noqa: E402
from eval.baseline.embed_store import (  # noqa: E402
    BaselineDenseStore,
    embed_texts,
    save_chunks,
)
from src.graph.retrieval import display_title  # noqa: E402

logging.getLogger("pdfminer").setLevel(logging.ERROR)


def _extract_pages(pdf_path: Path) -> list[str]:
    with pdfplumber.open(str(pdf_path)) as pdf:
        return [pg.extract_text() or "" for pg in pdf.pages]


def main():
    raw_dir = ROOT / "data" / "raw"
    out_dir = ROOT / "eval" / "data" / "baseline_index"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(p for p in raw_dir.iterdir()
                       if p.is_file() and p.suffix.lower() == ".pdf")
    print(f"found {len(pdf_files)} PDFs in {raw_dir}")

    all_chunks: list[dict] = []
    t0 = time.time()
    for pdf in pdf_files:
        title = display_title(pdf.name)
        try:
            pages = _extract_pages(pdf)
        except Exception as exc:
            print(f"  ! {pdf.name}: extract failed: {exc}")
            continue
        chunks = chunk_document(str(pdf), title, pages)
        for c in chunks:
            all_chunks.append(asdict(c))
        print(f"  - {pdf.name}: {len(pages)} pages -> {len(chunks)} chunks")

    print(f"\ntotal chunks: {len(all_chunks)} (extract took {time.time()-t0:.1f}s)")

    # Persist chunks.
    save_chunks(all_chunks, out_dir)

    # Build dense index.
    print("embedding chunks…")
    t1 = time.time()
    texts = [c["text"] for c in all_chunks]
    ids = [c["id"] for c in all_chunks]
    vecs = embed_texts(texts)
    dense = BaselineDenseStore(dim=vecs.shape[1])
    dense.add(ids, vecs)
    dense.save(out_dir)
    print(f"  dense index built in {time.time()-t1:.1f}s")

    # Build BM25.
    print("fitting BM25…")
    t2 = time.time()
    bm25 = BaselineBM25Store()
    bm25.fit(ids, texts)
    bm25.save(out_dir)
    print(f"  bm25 built in {time.time()-t2:.1f}s")

    # Manifest.
    manifest = {
        "n_pdfs": len(pdf_files),
        "n_chunks": len(all_chunks),
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "embedding_dim": int(vecs.shape[1]),
        "chunk_size_tokens": 500,
        "chunk_overlap_tokens": 100,
        "bm25_variant": "BM25Okapi",
        "fusion": "RRF k=60, dense_k=20, bm25_k=20, final_k=10",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nwrote manifest.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
