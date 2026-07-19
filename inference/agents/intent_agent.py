"Intent agent:Classify query -> search buckets + Chroma where."

from __future__ import annotations

from inference.agents.state import IntentState,Timer,merge_timing,GraphState
from inference.intent_classifier import ClassifiedQuery,classify_intent
from inference.retriever import build_where_filter


def _classified_query_to_state(classified:ClassifiedQuery)->IntentState:
    return{
        "intent_name":classified.intent.name,
        "retrieval_intent":classified.retrieval_intent,
        "search_types":list(classified.search_types),
        "confidence":classified.confidence,
        "where":build_where_filter(classified.search_types),
    }

def run_intent_agent(state:GraphState,*,query:str | None =None)->dict:
    """Classify one query into IntentState.
    -Single path :omit query-> uses retrieval query
    -Multi Path (later): passes query=sub_query"""

    text=(query or state.get("retrieval_query") or state.get("question") or "").strip()

    with Timer() as t:
        classified=classify_intent(text)

    return{
        "intent":_classified_query_to_state(classified),
        "timings":merge_timing(state, intent_ms=t.ms),
    }
