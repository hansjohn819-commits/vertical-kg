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

# 4b candidate filter (2026-05-06 收紧后):
#   keep iff cos >= MERGE_COS_HIGH  OR  (cos >= MERGE_COS_FLOOR AND jac >= MERGE_JACCARD_MIN)
# 同 type + cos≥0.55 那条分支已删——单一行业 corpus 下 type 信号被同质化稀释，
# 198 候选/161 节点的过松行为根因。Branch 1 强语义独证；Branch 2 中等语义 + 共邻居双证。
MERGE_TOP_K = 3            # embedding top-k per node（5→3：收紧后 branch 1 大头是单近邻 pass）
MERGE_JACCARD_MIN = 0.3    # neighbor-overlap threshold (Branch 2 双证之一)
MERGE_COS_HIGH = 0.85      # Branch 1 独证门槛
MERGE_COS_FLOOR = 0.70     # Branch 2 cos floor（与 jac AND 联用）
MERGE_MAX_ITER = 10        # hard cap on 4b convergence loops
MERGE_PAIR_CAP_PER_ROUND = 200  # §16.11 防 LLM judge 调用爆炸：候选过滤后按 cos 排序取 top-N

PRUNE_DELETE_AFTER = 3     # consecutive suspicious passes before delete
PRUNE_DOWNWEIGHT = 0.5     # weight multiplier on suspicious mark

# 4d link_form (2026-05-06 二次收紧):
LINK_BFS_DEPTH = 2         # BFS depth from each seed
LINK_MAX_NEW_PER_ROUND = 100  # safety net；正常工况靠 prompt + topology filter 控质量

# Bridge detection（替代旧的 LINK_CATEGORICAL_BRIDGE_TYPES type 列表）：
# 核心问题——type 列表是 domain-specific patches，新文档进来 M1 抽出新 type 又要补；
# 而 degree 阈值在冷启动失效（桥节点未合并前度数低）。新方案 multi-signal：
#
#   1. Edge-type containment pattern — 语言层 closed-set 语义信号，冷启动可用
#      (路径上所有边都是 is_part_of/located_in/contains/... 这种范畴 verb)
#   2. Degree hub threshold — 兜底机构 hub 类（FAO 这种，合并后度数自然涨）
#
# 任一信号触发即视为 bridge。type 列表彻底废弃。
LINK_HUB_DEGREE_THRESHOLD = 5  # in_deg + out_deg ≥ 5 视为 hub

SLEEP_PASS_LOG_PATH = "log.md"


# --- LangGraph state -----------------------------------------------------

class PassState(TypedDict, total=False):
    pass_id: str
    # merge-round state
    merge_iter: int
    merge_done_vote: bool
    merged_new_ids: Annotated[list[str], operator.add]
    # Per-pass set of pair keys (sorted "id1|id2") that the LLM judge
    # already said "different" about. Skipped on subsequent rounds —
    # candidate pool is deterministic in node ids so re-asking would
    # just burn the same answer.
    merge_rejected_pairs: Annotated[list[str], operator.add]
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
