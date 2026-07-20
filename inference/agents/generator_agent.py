"""Generator agent: retrieved hits → LoRA answer (remote)."""

from __future__ import annotations
import os
from inference.agents.state import GraphState,Timer,merge_timing
from inference.context import SYSTEM_PROMPT, format_multi_chunk_user_message
from inference.rag_engine import EMPTY_ANSWER,_generate_remote

def _gen_query(state:GraphState)-> str:
    return (state.get("retrieval_query") or state.get("question") or "").strip()


def run_generator_agent(state:GraphState,*,inference_url:str | None =None)->dict:
    # Cache hit already populated answer — skip GPU

    if state.get("cache_hit") and (state.get("answer") or "").strip():
        return {}

    hits=state.get("hits") or []
    if not hits:
        return{
            "answer":EMPTY_ANSWER,
            "cache_hit":False,
        }

    query=_gen_query(state)
    url= (inference_url or os.environ.get("INFERENCE_URL") or "").rstrip("/")
    if not url:
        raise RuntimeError("Inference URL not configured. start runpod/colab and set .env")
    
    user_content= format_multi_chunk_user_message(query,hits)
    messages=[
        {"role":"system", "content":SYSTEM_PROMPT},
        {"role":"user", "content":user_content},
    ]

    max_new_tokens=int(state.get("max_new_tokens") or 512)
    temperature=float(state.get("temperature") or 0.2)
    with Timer() as t:
        answer=_generate_remote(
            messages,
            url=url,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
    return {
        "answer":answer,
        "timings":merge_timing(state, generate_ms=t.ms),
    }
