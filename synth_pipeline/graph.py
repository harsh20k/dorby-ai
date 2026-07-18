"""Linear LangGraph: seed → generate → filter ⇄ retry → judge → staging."""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

from synth_pipeline.config import PipelineConfig
from synth_pipeline.nodes.filter import heuristic_filter_node
from synth_pipeline.nodes.generate import generate_node
from synth_pipeline.nodes.judge import judge_node
from synth_pipeline.nodes.seed import seed_sampler_node
from synth_pipeline.nodes.writer import staging_node
from synth_pipeline.state import PairState


def _route_after_filter(
    state: PairState, *, max_retries: int
) -> Literal["generate", "semantic_judge", "staging"]:
    qc = state.get("qc") or {}
    if qc.get("filter_passed"):
        return "semantic_judge"
    if int(state.get("retry_count", 0)) <= max_retries:
        return "generate"
    return "staging"


def _route_after_judge(state: PairState) -> Literal["staging"]:
    del state
    return "staging"


def build_graph(cfg: PipelineConfig):
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "langgraph is required. pip install -r requirements.txt"
        ) from exc

    graph = StateGraph(PairState)
    graph.add_node("seed_sampler", partial(seed_sampler_node, cfg=cfg))
    graph.add_node("generate", partial(generate_node, cfg=cfg))
    graph.add_node("heuristic_filter", partial(heuristic_filter_node, cfg=cfg))
    graph.add_node("semantic_judge", partial(judge_node, cfg=cfg))
    graph.add_node("staging", partial(staging_node, cfg=cfg))

    graph.set_entry_point("seed_sampler")
    graph.add_edge("seed_sampler", "generate")
    graph.add_edge("generate", "heuristic_filter")
    graph.add_conditional_edges(
        "heuristic_filter",
        partial(_route_after_filter, max_retries=cfg.max_retries),
        {
            "generate": "generate",
            "semantic_judge": "semantic_judge",
            "staging": "staging",
        },
    )
    graph.add_conditional_edges(
        "semantic_judge",
        _route_after_judge,
        {"staging": "staging"},
    )
    graph.add_edge("staging", END)
    return graph.compile()


def run_one(cfg: PipelineConfig, initial: PairState) -> dict[str, Any]:
    app = build_graph(cfg)
    # LangSmith: set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY.
    return app.invoke(initial)
