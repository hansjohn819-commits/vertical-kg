"""Shared state + constants for the M4 pass (guide §5).

LangGraph state is a TypedDict; list fields use Annotated + operator.add so
per-step updates accumulate instead of overwriting.
"""

import operator
from typing import Annotated, Any, TypedDict

# --- P1 defaults (see guide §5.4 / §5.5 / §5.6 / §5.7) -------------------

DELTA_TRAVERSE = 0.01      # edge weight bump per Q&A traversal
DELTA_CITE = 0.02          # extra bump for the seed/cited nodes' edges
DECAY = 0.95               # global edge-weight decay per pass

MERGE_TOP_K = 5            # embedding top-k per node for 4b candidates
MERGE_JACCARD_MIN = 0.3    # neighbor-overlap threshold (strong topology signal)
MERGE_COS_MIN = 0.85       # embedding cos threshold (strong semantic signal)
MERGE_COS_SAME_TYPE = 0.55 # same-type pairs pass with a looser cos (guide §5.5 "同类型" branch)
MERGE_MAX_ITER = 10        # hard cap on 4b convergence loops

PRUNE_DELETE_AFTER = 3     # consecutive suspicious passes before delete
PRUNE_DOWNWEIGHT = 0.5     # weight multiplier on suspicious mark

LINK_BFS_DEPTH = 2         # 4d BFS depth
LINK_MAX_NEW_PER_ROUND = 20  # safety cap per 4d iteration

SLEEP_PASS_LOG_PATH = "log.md"


# --- LangGraph state -----------------------------------------------------

class PassState(TypedDict, total=False):
    pass_id: str
    # merge-round state
    merge_iter: int
    merge_done_vote: bool
    merged_new_ids: Annotated[list[str], operator.add]
    # prune-round state
    prune_iter: int
    prune_changed: bool
    # link-round state
    link_iter: int
    link_changed: bool
    link_tried_pairs: list[str]  # replaces (not appends) each round
    seeded_for_link: Annotated[list[str], operator.add]   # merge/reinforce outputs
    # cumulative
    stats: dict[str, Any]
    events: Annotated[list[dict], operator.add]
