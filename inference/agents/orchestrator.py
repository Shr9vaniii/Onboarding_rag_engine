"""Sequential orchestrator: chains agents (kept as fallback / reference).

Prefer ``inference.agents.graph.ask_graph`` for the LangGraph path.
"""

from __future__ import annotations

import time

from inference.agents.assembler_agent import run_assembler_agent
from inference.agents.cache_agent import run_cache_get, run_cache_set
from inference.agents.conversation_agent import run_conversation_agent  # note: your current spelling
from inference.agents.generator_agent import run_generator_agent
from inference.agents.hybrid_retrieval_agent import run_hybrid_retrieval_agent
from inference.agents.intent_agent import run_intent_agent
from inference.agents.query_decomposer_agent import run_decompose_agent
from inference.agents.state import GraphState, SubQueryState, Timer, initial_state, merge_timing
from inference.cache_store import AnswerCache
from inference.session_store import SessionStore


def _run_single_arm(
    state: GraphState,
    *,
    query: str,
    answer_cache: AnswerCache,
    inference_url: str | None,
) -> SubQueryState:
    """Intent → Hybrid → Cache → Generator → Cache.set for one query."""
    arm = dict(state)
    arm.update(run_intent_agent(arm, query=query))
    arm.update(run_hybrid_retrieval_agent(arm, query=query))
    arm.update(run_cache_get(arm, answer_cache=answer_cache))
    try:
        arm.update(run_generator_agent(arm, inference_url=inference_url))
        arm.update(run_cache_set(arm, answer_cache=answer_cache))
        error = ""
    except Exception as exc:
        error = str(exc)
        arm["answer"] = (
            "**Direct Answer:**\n"
            "I couldn't generate an answer for this part due to a temporary error.\n\n"
            f"**Error:** {exc}"
        )
        arm["partial"] = True

    hits = list(arm.get("hits") or [])
    return {
        "query": query,
        "intent": arm.get("intent") or {},
        "hits": hits,
        "answer": arm.get("answer") or "",
        "num_chunks": len(hits),
        "cache_hit": bool(arm.get("cache_hit")),
        "error": error,
    }


def run_orchestrator(
    state: GraphState,
    *,
    session_store: SessionStore,
    answer_cache: AnswerCache,
    inference_url: str | None = None,
) -> GraphState:
    """
    Conversation → Decompose →
      multi: loop arms → Assembler
      single: Intent→Hybrid→Cache→Generator→Cache→Assembler
    """
    started = time.perf_counter()

    state.update(run_conversation_agent(state, session_store=session_store))
    state.update(run_decompose_agent(state))

    if state.get("is_multi"):
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
                errors.append(arm["error"])
            sub_results.append(arm)

        state["sub_results"] = sub_results
        state["partial"] = partial
        state["errors"] = errors
        # clear single-path fields; Assembler uses sub_results
        state["hits"] = []
        state["answer"] = ""
    else:
        q = (state.get("retrieval_query") or state.get("question") or "").strip()
        state.update(run_intent_agent(state, query=q))
        state.update(run_hybrid_retrieval_agent(state, query=q))
        state.update(run_cache_get(state, answer_cache=answer_cache))
        state.update(run_generator_agent(state, inference_url=inference_url))
        state.update(run_cache_set(state, answer_cache=answer_cache))

    state.update(run_assembler_agent(state, session_store=session_store))
    state["timings"] = merge_timing(
        state,
        total_ms=(time.perf_counter() - started) * 1000,
    )
    return state


def ask(
    question: str,
    *,
    session_store: SessionStore,
    answer_cache: AnswerCache | None = None,
    session_id: str = "",
    use_cache: bool = True,
    inference_url: str | None = None,
    **kwargs,
) -> GraphState:
    """Convenience: build initial state + run orchestrator."""
    state = initial_state(
        question,
        session_id=session_id,
        use_cache=use_cache,
        **kwargs,
    )
    cache = answer_cache if answer_cache is not None else AnswerCache(enabled=use_cache)
    return run_orchestrator(
        state,
        session_store=session_store,
        answer_cache=cache,
        inference_url=inference_url,
    )