"""LangGraph StateGraph for the full 4c -> 4b -> 4a -> 4d pass (guide §5.0).

Each sub-operation is a node; the three loops are modeled as
conditional edges that route either back to the same node (continue the
loop) or forward to the next sub-operation (converged).

4c is one-shot (deterministic, no LLM). 4b converges on an LLM done vote
(with MERGE_MAX_ITER hard cap). 4a / 4d converge mechanically on "no
change this round".
"""

from functools import partial

from langgraph.graph import END, START, StateGraph

from src.graph.instance import GraphInstance

from .link_form import link_should_continue, link_step
from .merge import merge_should_continue, merge_step
from .prune import prune_should_continue, prune_step
from .reinforce import reinforce_step
from .state import PassState


def build_pass_graph(instance: GraphInstance):
    g: StateGraph = StateGraph(PassState)

    g.add_node("reinforce", partial(reinforce_step, instance=instance))
    g.add_node("merge", partial(merge_step, instance=instance))
    g.add_node("prune", partial(prune_step, instance=instance))
    g.add_node("link", partial(link_step, instance=instance))
    g.add_node("finalize", _finalize)

    g.add_edge(START, "reinforce")
    g.add_edge("reinforce", "merge")
    g.add_conditional_edges("merge", merge_should_continue, {"merge": "merge", "prune": "prune"})
    g.add_conditional_edges("prune", prune_should_continue, {"prune": "prune", "link": "link"})
    g.add_conditional_edges("link", link_should_continue, {"link": "link", "finalize": "finalize"})
    g.add_edge("finalize", END)

    return g.compile()


def _finalize(state: PassState) -> dict:
    # Compile-time no-op — stats are already in state; runner saves storage
    # after the graph completes.
    return {}
