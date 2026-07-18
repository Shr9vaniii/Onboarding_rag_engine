"""
Intent-aware retrieval over ChromaDB v3 using hybrid dense + BM25 + rerank.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.context import resolve_chunk_type
from inference.intent_classifier import ClassifiedQuery, classify_intent
from retrieval.hybrid_retrieval import hybrid_retrieve

DEFAULT_DUAL_CONFIDENCE_THRESHOLD = 0.75


def chunk_type_to_where(chunk_type: str) -> dict:
    if chunk_type == "reference":
        return {"$and": [{"type": "wikis_arch_docs"}, {"subtype": "reference"}]}
    if chunk_type == "wikis_arch_docs":
        return {"$and": [{"type": "wikis_arch_docs"}, {"subtype": {"$ne": "reference"}}]}
    return {"type": chunk_type}


def build_where_filter(search_types: list[str]) -> dict | None:
    if not search_types:
        return None
    if len(search_types) == 1:
        return chunk_type_to_where(search_types[0])
    return {"$or": [chunk_type_to_where(t) for t in search_types]}


def should_dual_retrieve(
    *,
    needs_rewrite: bool,
    confidence: float,
    topic_status: str,
    original: str,
    retrieval_query: str,
    threshold: float = DEFAULT_DUAL_CONFIDENCE_THRESHOLD,
) -> bool:
    """Run both original and rewritten queries when rewrite confidence is low."""
    if not needs_rewrite:
        return False
    if original.strip().lower() == retrieval_query.strip().lower():
        return False
    if topic_status == "ambiguous":
        return True
    return confidence < threshold


def merge_results_rrf(
    result_lists: list[list[dict]],
    *,
    n_final: int,
    k: int = 60,
) -> list[dict]:
    """Merge multiple ranked result lists with reciprocal rank fusion."""
    scores: dict[str, float] = {}
    id_to_hit: dict[str, dict] = {}

    for results in result_lists:
        for rank, hit in enumerate(results):
            doc_id = hit.get("id")
            if not doc_id:
                meta = hit.get("metadata") or {}
                doc_id = str(meta.get("source", "")) or f"rank-{rank}"
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            existing = id_to_hit.get(doc_id)
            if existing is None or hit.get("final_score", 0) > existing.get("final_score", 0):
                id_to_hit[doc_id] = hit

    merged: list[dict] = []
    for doc_id in sorted(scores, key=scores.get, reverse=True)[:n_final]:
        hit = dict(id_to_hit[doc_id])
        hit["dual_rrf_score"] = scores[doc_id]
        merged.append(hit)
    return merged


def retrieve_dual(
    original_query: str,
    rewritten_query: str,
    *,
    n_final: int = 5,
    rerank: bool = True,
    verbose: bool = False,
    confidence_threshold: float = DEFAULT_DUAL_CONFIDENCE_THRESHOLD,
) -> tuple[ClassifiedQuery, list[dict]]:
    """Retrieve with both queries and merge via RRF."""
    classified, primary_results = retrieve_for_query(
        rewritten_query,
        n_final=n_final,
        rerank=rerank,
        verbose=False,
    )
    _, original_results = retrieve_for_query(
        original_query,
        n_final=n_final,
        rerank=rerank,
        verbose=False,
    )
    merged = merge_results_rrf([primary_results, original_results], n_final=n_final)

    if verbose:
        print(
            f"Dual retrieval: merged original + rewritten "
            f"(confidence < {confidence_threshold:.2f} or ambiguous topic)"
        )
        print(f"  Original:  {original_query!r}")
        print(f"  Rewritten: {rewritten_query!r}")
        for i, hit in enumerate(merged, 1):
            meta = hit.get("metadata") or {}
            print(
                f"  [{i}] {hit.get('chunk_type', '?')} | "
                f"dual_rrf={hit.get('dual_rrf_score', 0):.4f} "
                f"final={hit.get('final_score', hit.get('rerank_score', 0)):.3f} | "
                f"{str(meta.get('source', ''))[:60]}"
            )

    return classified, merged


def retrieve_multi(
    queries: list[str],
    *,
    n_final: int = 5,
    rerank: bool = True,
    verbose: bool = False,
) -> list[tuple[ClassifiedQuery, list[dict]]]:
    """Retrieve independently for each sub-query (no merge)."""
    outputs: list[tuple[ClassifiedQuery, list[dict]]] = []
    for i, query in enumerate(queries, 1):
        if verbose:
            print(f"\n--- Sub-query [{i}/{len(queries)}] ---")
        classified, results = retrieve_for_query(
            query,
            n_final=n_final,
            rerank=rerank,
            verbose=verbose,
        )
        outputs.append((classified, results))
    return outputs


def retrieve_for_query(
    query: str,
    *,
    n_final: int = 5,
    rerank: bool = True,
    verbose: bool = False,
) -> tuple[ClassifiedQuery, list[dict]]:
    classified = classify_intent(query)
    where = build_where_filter(classified.search_types)

    if verbose:
        print(f"Query:     {classified.original}")
        print(f"Intent:    {classified.intent.name} ({classified.confidence})")
        print(f"Searching: {classified.search_types}")
        print(f"Where:     {where}")

    results = hybrid_retrieve(
        query=classified.original,
        intent=classified.retrieval_intent,
        n_dense=10,
        n_bm25=10,
        n_final=n_final,
        where=where,
        rerank=rerank,
    )

    # Low-confidence fallback — search all main buckets
    if classified.confidence == "low" and not results:
        if verbose:
            print("  No results — broadening to all types")
        results = hybrid_retrieve(
            query=classified.original,
            intent="wikis_arch_docs",
            n_final=n_final,
            where=None,
            rerank=rerank,
        )

    for hit in results:
        meta = hit.get("metadata") or {}
        hit["chunk_type"] = resolve_chunk_type(meta)

    if classified.search_types:
        allowed = set(classified.search_types)
        typed = [h for h in results if h.get("chunk_type") in allowed]
        if typed:
            results = typed

    if verbose:
        for i, hit in enumerate(results, 1):
            meta = hit.get("metadata") or {}
            print(
                f"  [{i}] {hit.get('chunk_type', '?')} | "
                f"final={hit.get('final_score', hit.get('rerank_score', 0)):.3f} "
                f"(rerank={hit.get('rerank_score', 0):.3f} boost={hit.get('entity_boost', 0):+.1f}) | "
                f"{str(meta.get('source', ''))[:60]}"
            )

    return classified, results


def top_chunk(results: list[dict]) -> dict | None:
    return results[0] if results else None
