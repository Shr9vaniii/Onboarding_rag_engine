"Decomposer Agent."

from __future__ import annotations
from inference.agents.state import DecomposeState,Timer,merge_timing,GraphState
from inference.query_decomposer import DecomposeResult,decompose_query

def _decompose_to_state(decompose:DecomposeResult)->DecomposeState:
    return{
        "is_multi":decompose.is_multi,
        "sub_queries":list(decompose.sub_queries),
        "confidence":decompose.confidence,
        "reason":decompose.reason,
    }

def run_decompose_agent(state:GraphState)->dict:
    query=(state.get("retrieval_query") or state.get("question") or "").strip()
    with Timer() as t:
        result=decompose_query(query)

    return{
        "decompose":_decompose_to_state(result),
        "sub_queries":list(result.sub_queries),
        "is_multi":result.is_multi,
        "timings":merge_timing(state, decompose_ms=t.ms),
    }