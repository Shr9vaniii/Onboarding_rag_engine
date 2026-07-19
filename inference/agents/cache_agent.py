"Cache agent: Redis get/set after hybrid retrieval."

from __future__ import annotations

from inference.agents.state import GraphState,Timer,merge_timing
from inference.cache_store import AnswerCache

def _cache_query(state:GraphState)->str:
    return (state.get("retrieval_query") or state.get("question") or "").strip()


def run_cache_get(state:GraphState,*,answer_cache:AnswerCache)->dict:
    """After Hybrid. On HIT, Assembler/Send can skip Generator."""
    if not state.get("use_cache", True):
        return {"cache_hit":False}
    hits=state.get("hits") or []

    if not hits:
        return {"cache_hit":False}
    

    query=_cache_query(state)
    with Timer() as t:
        cached= answer_cache.get(query,hits)
    
    if cached:
        return{
            "cache_hit":True,
            "answer":cached,
            "timings":merge_timing(state, cache_ms=t.ms),
        }
    return {
        "cache_hit":False,
        "timings":merge_timing(state, cache_ms=t.ms),
    }

def run_cache_set(state:GraphState,*,answer_cache:AnswerCache)->dict:
    """After Generator (miss path only). Abstentions are skipped inside AnswerCache.set."""

    if not state.get("use_cache", True):
        return {}
    if state.get("cache_hit"):
        return {}

    hits=state.get("hits") or []
    answer=state.get("answer") or ""
    if not hits or not answer:
        return {}
    
    query=_cache_query(state)
    with Timer() as t:
        answer_cache.set(query,hits,answer)
    return {
        "timings":merge_timing(state, cache_ms=t.ms),
    }