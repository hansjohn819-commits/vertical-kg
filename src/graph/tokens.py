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

# Ingest input cap (guide §12.5.4 M1).
INGEST_INPUT_MAX_TOKENS = 25_000

# M3 integrate soft cap — online Streamlit paste path (guide §12.5.4 M3).
# Over this, the integrator LLM-compresses the input before extraction
# (guide §12.5.5 "agent 自压缩" direction). Not a hard reject.
INTEGRATE_INPUT_SOFT_CAP_TOKENS = 8_000


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def within(text: str, cap: int) -> bool:
    return count_tokens(text) <= cap
