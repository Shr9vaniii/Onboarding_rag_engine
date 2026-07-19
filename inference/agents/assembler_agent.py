"""Assembler agent: citations + multi stitch + session persist."""

from __future__ import annotations

from inference.agents.state import CitationState, GraphState, SubQueryState
from inference.rag_engine import PARTIAL_FAILURE_PREFIX
from inference.session_store import SessionStore


def _pack_hits(hits: list[dict]) -> tuple[list[str], list[str], list[float], list[CitationState]]:
    sources = [str((h.get("metadata") or {}).get("source", "")) for h in hits]
    chunk_types = [h.get("chunk_type", "") for h in hits]
    scores = [
        float(h.get("final_score", h.get("rerank_score", h.get("rrf_score", 0))))
        for h in hits
    ]
    citations: list[CitationState] = []
    for h in hits:
        meta = h.get("metadata") or {}
        citations.append(
            {
                "id": str(h.get("id", "")),
                "source": str(meta.get("source", "")),
                "name": str(meta.get("name") or meta.get("qualified_name", "")),
                "chunk_type": str(h.get("chunk_type", "")),
                "score": float(
                    h.get("final_score", h.get("rerank_score", h.get("rrf_score", 0)))
                ),
                "snippet": (h.get("content") or "")[:240],
            }
        )
    return sources, chunk_types, scores, citations


def _format_multi(sub_results: list[SubQueryState], *, partial: bool) -> str:
    parts: list[str] = []
    if partial:
        parts.append(PARTIAL_FAILURE_PREFIX.strip())
    for i, sr in enumerate(sub_results, 1):
        q = sr.get("query") or f"Part {i}"
        a = sr.get("answer") or ""
        parts.append(f"### Question {i}: {q}\n\n{a}")
    return "\n\n---\n\n".join(parts)


def run_assembler_agent(
    state: GraphState,
    *,
    session_store: SessionStore | None = None,
) -> dict:
    sub_results = list(state.get("sub_results") or [])
    partial = bool(state.get("partial"))
    errors = list(state.get("errors") or [])

    # --- Multi path ---
    if state.get("is_multi") and sub_results:
        all_sources: list[str] = []
        all_types: list[str] = []
        all_scores: list[float] = []
        all_citations: list[CitationState] = []
        any_cache = False
        for sr in sub_results:
            hits = list(sr.get("hits") or [])
            sources, types, scores, citations = _pack_hits(hits)
            # Prefer per-arm packed fields if already set
            all_sources.extend(sr.get("sources") or sources)
            all_types.extend(sr.get("chunk_types") or types)
            all_scores.extend(sr.get("retrieval_scores") or scores)
            all_citations.extend(sr.get("citations") or citations)
            any_cache = any_cache or bool(sr.get("cache_hit"))
            if sr.get("error"):
                partial = True
                errors.append(str(sr["error"]))

        answer = _format_multi(sub_results, partial=partial)
        update = {
            "answer": answer,
            "citations": all_citations,
            "sources": all_sources,
            "chunk_types": all_types,
            "retrieval_scores": all_scores,
            "num_chunks": sum(int(sr.get("num_chunks") or len(sr.get("hits") or [])) for sr in sub_results),
            "partial": partial,
            "errors": errors,
            "cache_hit": any_cache,
        }
    else:
        # --- Single path ---
        hits = list(state.get("hits") or [])
        sources, types, scores, citations = _pack_hits(hits)
        update = {
            "citations": citations,
            "sources": sources,
            "chunk_types": types,
            "retrieval_scores": scores,
            "num_chunks": len(hits),
            "partial": partial,
            "errors": errors,
            # keep existing answer from Generator / Cache
        }

    # Persist conversational turn
    if (
        session_store is not None
        and state.get("persist_session", True)
        and (state.get("session_id") or "").strip()
    ):
        session_id = state["session_id"].strip()
        question = state.get("question") or ""
        answer = update.get("answer") or state.get("answer") or ""
        if question and answer:
            session_store.add_turn(session_id, question, answer)

    return update