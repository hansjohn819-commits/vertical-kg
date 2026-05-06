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

# PDF chunking overlap: each chunk after the first repeats this many pages
# from the tail of the previous chunk so cross-page sentences and tables
# aren't sliced (guide §12.5.5).
INGEST_PDF_PAGE_OVERLAP = 2


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def within(text: str, cap: int) -> bool:
    return count_tokens(text) <= cap
