"""Token counting + budget constants.

Uses tiktoken's cl100k_base as a rough estimator. It is NOT Gemma's real
tokenizer — may under-count by 10–20% on Chinese/symbols. Per guide §12.5.1,
apply `SAFETY_FACTOR` to any budget comparison.
"""

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")

# Apply this factor when comparing tiktoken estimates against the real
# context window. Real-token-count ≈ tiktoken-count × SAFETY_FACTOR.
SAFETY_FACTOR = 1.2

# Node-field hard caps (tiktoken estimated), per guide §12.5.3.
SUMMARY_MAX_TOKENS = 300
DETAIL_MAX_TOKENS = 8_000

# Ingest input cap (guide §12.5.4 M1). Sized for the 128K context window:
# 80K tiktoken input × 1.2 safety ≈ 96K real input + ~10K extraction output
# + ~1K prompt overhead ≈ 107K real, comfortably under 128K.
INGEST_INPUT_MAX_TOKENS = 80_000

# PDF chunking overlap: kept for back-compat of any caller still using the
# constant, but §16.17 cut over to per-page chunking so the overlap is now
# 0 — chunk = exactly one page, no sliding window. Cross-page coreference
# is handled at the LLM-prompt layer instead (stateful prior_entities block
# in M1 PASS 1, see §16.17.3).
INGEST_PDF_PAGE_OVERLAP = 0

# Super-chunk wrapper above M1 (2026-05-10). Long PDFs get split into
# super-chunks of at most this many pages BEFORE handing off to M1; each
# super-chunk runs the full M1 pipeline (PASS 1 + PASS 2 + intra-doc fuse
# + reclassifier + dedup) as if it were its own document. Bounded so
# PASS 1's `prior_entities` block can never grow large enough to overflow
# the 128K context window: 80 pages × ~8 entities/page × ~84 real tokens
# per prior-block line ≈ 54K — leaves comfortable headroom for ontology,
# system prompt, page text, and output. Cross-super-chunk same-entity
# duplicates are accepted at ingest time and resolved by M4b sleep pass
# (the same path that handles cross-document duplicates).
INGEST_SUPERCHUNK_MAX_PAGES = 80
# 1-page overlap between consecutive super-chunks so an entity defined on
# the boundary page isn't lost when it's first mentioned in one super-
# chunk and referenced again at the start of the next.
INGEST_SUPERCHUNK_OVERLAP_PAGES = 1


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def within(text: str, cap: int) -> bool:
    return count_tokens(text) <= cap
