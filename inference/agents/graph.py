"""LangGraph wiring for the onboarding RAG agents.

Phase 2: same agents as the sequential orchestrator, now as graph nodes.
Nodes still wrap existing tools (rewriter, hybrid retrieve, LoRA, Redis).
"""

from __future__ import annotations

import time
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from inference.agents.assembler_agent import run_assembler_agent
from inference.agents.cache_agent import run_cache_get, run_cache_set
from inference.agents.conversation_agent import run_conversation_agent
from inference.agents.generator_agent import run_generator_agent
from inference.agents.hybrid_retrieval_agent import run_hybrid_retrieval_agent
from inference.agents.intent_agent import run_intent_agent
from inference.agents.orchestrator import _run_single_arm
from inference.agents.query_decomposer_agent import run_decompose_agent
from inference.agents.state import GraphState, SubQueryState, initial_state, merge_timing
from inference.cache_store import AnswerCache
from inference.session_store import SessionStore


def _route_after_decompose(state: GraphState) -> Literal["multi_arms", "intent"]:
    return "multi_arms" if state.get("is_multi") else "intent"


def build_rag_graph(
    *,
    session_store: SessionStore,
    answer_cache: AnswerCache,
    inference_url: str | None = None,
):
    """Compile a StateGraph. Inject stores so nodes stay thin wrappers."""

    def conversation(state: GraphState) -> dict[str, Any]:
        return run_conversation_agent(state, session_store=session_store)

    def decompose(state: GraphState) -> dict[str, Any]:
        return run_decompose_agent(state)

    def intent(state: GraphState) -> dict[str, Any]:
        q = (state.get("retrieval_query") or state.get("question") or "").strip()
        return run_intent_agent(state, query=q)

    def hybrid(state: GraphState) -> dict[str, Any]:
        q = (state.get("retrieval_query") or state.get("question") or "").strip()
        return run_hybrid_retrieval_agent(state, query=q)

    def cache_get(state: GraphState) -> dict[str, Any]:
        return run_cache_get(state, answer_cache=answer_cache)

    def generate(state: GraphState) -> dict[str, Any]:
        return run_generator_agent(state, inference_url=inference_url)

    def cache_set(state: GraphState) -> dict[str, Any]:
        return run_cache_set(state, answer_cache=answer_cache)

    def multi_arms(state: GraphState) -> dict[str, Any]:
        """Fan-out loop as one node (Send API can come later)."""
        sub_results: list[SubQueryState] = []
        partial = False
        errors: list[str] = []
        for q in state.get("sub_queries") or []:
            arm = _run_single_arm(
                state,
                query=q,
                answer_cache=answer_cache,
                inference_url=inference_url,
            )
            if arm.get("error"):
                partial = True
                errors.append(str(arm["error"]))
            sub_results.append(arm)
        return {
            "sub_results": sub_results,
            "partial": partial,
            "errors": errors,
            "hits": [],
            "answer": "",
        }

    def assembler(state: GraphState) -> dict[str, Any]:
        return run_assembler_agent(state, session_store=session_store)

    def finalize(state: GraphState) -> dict[str, Any]:
        # total_ms is set by ask_graph around invoke; keep node for future hooks
        return {}

    g = StateGraph(GraphState)

    g.add_node("conversation", conversation)
    g.add_node("decompose", decompose)
    g.add_node("intent", intent)
    g.add_node("hybrid", hybrid)
    g.add_node("cache_get", cache_get)
    g.add_node("generate", generate)
    g.add_node("cache_set", cache_set)
    g.add_node("multi_arms", multi_arms)
    g.add_node("assembler", assembler)
    g.add_node("finalize", finalize)

    g.add_edge(START, "conversation")
    g.add_edge("conversation", "decompose")
    g.add_conditional_edges(
        "decompose",
        _route_after_decompose,
        {
            "multi_arms": "multi_arms",
            "intent": "intent",
        },
    )

    # Single path
    g.add_edge("intent", "hybrid")
    g.add_edge("hybrid", "cache_get")
    g.add_edge("cache_get", "generate")
    g.add_edge("generate", "cache_set")
    g.add_edge("cache_set", "assembler")

    # Multi path
    g.add_edge("multi_arms", "assembler")

    g.add_edge("assembler", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


def ask_graph(
    question: str,
    *,
    session_store: SessionStore,
    answer_cache: AnswerCache | None = None,
    session_id: str = "",
    use_cache: bool = True,
    inference_url: str | None = None,
    **kwargs,
) -> GraphState:
    """Same API as orchestrator.ask, but runs the LangGraph."""
    state = initial_state(
        question,
        session_id=session_id,
        use_cache=use_cache,
        **kwargs,
    )
    cache = answer_cache if answer_cache is not None else AnswerCache(enabled=use_cache)
    graph = build_rag_graph(
        session_store=session_store,
        answer_cache=cache,
        inference_url=inference_url,
    )
    started = time.perf_counter()
    result = graph.invoke(state)
    result["timings"] = merge_timing(
        result,  # type: ignore[arg-type]
        total_ms=(time.perf_counter() - started) * 1000,
    )
    return result  # type: ignore[return-value]
