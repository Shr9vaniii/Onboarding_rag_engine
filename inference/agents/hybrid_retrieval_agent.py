"Hybrid retrieval agent: intent.where -> dense+BM25+rerank hits."

from __future__ import annotations
from inference.agents.state import GraphState,Timer,merge_timing
from inference.context import resolve_chunk_type
from inference.retriever import merge_results_rrf,should_dual_retrieve
from retrieval.hybrid_retrieval import hybrid_retrieve

def _attach_chunk_types(hits:list[dict])->list[dict]:
    for hit in hits:
        meta=hit.get("metadata") or {}
        hit["chunk_type"]=resolve_chunk_type(meta)

    return hits

def _filter_by_search_types(hits:list[dict],search_types:list[str])->list[dict]:
    if not search_types:
        return hits
    allowed=set(search_types)
    typed=[h for h in hits if h.get("chunk_type") in allowed]
    return typed or hits

def _run_hybrid(
    query:str,
    *,
    retrieval_intent: str,
    where:dict | None,
    top_k:int,
    search_types:list[str],

)->list[dict]:
    hits=hybrid_retrieve(
        query=query,
        intent=retrieval_intent,
        n_dense=10,
        n_bm25=10,
        n_final=top_k,
        where=where,
        rerank=True,
    )

    hits=_attach_chunk_types(hits)
    return _filter_by_search_types(hits,search_types)

def run_hybrid_retrieval_agent(state:GraphState,*,query:str | None =None)->dict:
    """
    Single path: omit query ->uses retrieval query / question+state[inetnt].
    Multi path (later): pass subquery and ensure intent was set for that arm.
    """

    text=(query or state.get("retrieval_query") or state.get("question") or "").strip()
    intent=state.get("intent") or {}
    top_k=int(state.get("top_k") or 5)

    retrieval_intent=intent.get("retrieval_intent") or ""
    where=intent.get("where") or None
    search_types=list(intent.get("search_types") or [])

    with Timer() as t:
        rewrite= state.get("rewrite") or {}
        use_dual=bool(rewrite and should_dual_retrieve(
            needs_rewrite=bool(rewrite.get("needs_rewrite")),
            confidence=float(rewrite.get("confidence") or 0.0),
            topic_status=str(rewrite.get("topic_status") or "same"),
            original=state.get("question") or text,
            retrieval_query=text,
        ))

        if use_dual:
            primary= _run_hybrid(
                text,
                retrieval_intent=retrieval_intent,
                where=where,
                top_k=top_k,
                search_types=search_types,
            )

            #second arm :original user question (intent for rewritten query still used)

            secondary= _run_hybrid(
                state.get("question") or text,
                retrieval_intent=retrieval_intent,
                where=where,
                top_k=top_k,
                search_types=search_types,
            )

            hits=merge_results_rrf([primary,secondary],n_final=top_k)
        else:

            hits= _run_hybrid(
                text,
                retrieval_intent=retrieval_intent,
                where=where,
                top_k=top_k,
                search_types=search_types,
            )

            #low confidence empty fallback (same as retriever,retrieve_for_query)
            if not hits and intent.get("confidence") == "low":
                hits= _run_hybrid(
                    text,
                    retrieval_intent="wikis_arch_docs",
                    where=None,
                    top_k=top_k,
                    search_types=[],
                )

        return {
            "hits":hits,
            "dual_retrieval":use_dual,
            "timings":merge_timing(state, retrieve_ms=t.ms),
        }