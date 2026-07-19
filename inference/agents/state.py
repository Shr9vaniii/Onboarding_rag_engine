"""Shared graph state for the multi-agent RAG pipeline.

LangGraph-ready: each agent returns a partial update into GraphState.
Intent and hybrid retrieval are separate fields / future nodes.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, TypedDict

# ---------------------------------------------------------------------------
# Nested shapes (plain dicts so the graph stays JSON-serializable-friendly)
# ---------------------------------------------------------------------------


class TimingState(TypedDict, total=False):
    rewrite_ms: float
    decompose_ms: float
    intent_ms: float
    retrieve_ms: float
    cache_ms: float
    generate_ms: float
    verify_ms: float
    total_ms: float


class RewriteState(TypedDict, total=False):
    needs_rewrite: bool
    standalone_query: str
    topic_status: str
    entities: list[str]
    confidence: float
    reason: str
    active_topic: str
    topic_summary: str


class DecomposeState(TypedDict, total=False):
    is_multi: bool
    sub_queries: list[str]
    confidence: float
    reason: str


class IntentState(TypedDict, total=False):
    """Output of IntentAgent (separate from hybrid retrieval)."""

    intent_name: str
    retrieval_intent: str
    search_types: list[str]
    confidence: str
    where: dict[str, Any] | None


class CitationState(TypedDict, total=False):
    id: str
    source: str
    name: str
    chunk_type: str
    score: float
    snippet: str


class SubQueryState(TypedDict, total=False):
    """One arm of the multi-query loop: Intent → Hybrid → Cache → Generator."""

    query: str
    intent: IntentState
    hits: list[dict[str, Any]]
    answer: str
    citations: list[CitationState]
    sources: list[str]
    chunk_types: list[str]
    retrieval_scores: list[float]
    num_chunks: int
    cache_hit: bool
    error: str


# ---------------------------------------------------------------------------
# Graph state (passed through Conversation → Planner → Intent → Hybrid → …)
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    # Request / session
    request_id: str
    session_id: str
    question: str

    # ConversationAgent
    retrieval_query: str
    rewrite: RewriteState | None

    # PlannerAgent
    decompose: DecomposeState | None
    sub_queries: list[str]
    is_multi: bool

    # IntentAgent (single-path; multi uses sub_results[].intent)
    intent: IntentState | None

    # HybridRetrievalAgent (single-path; multi uses sub_results[].hits)
    hits: list[dict[str, Any]]
    dual_retrieval: bool

    # CacheAgent + GeneratorAgent (single-path)
    cache_hit: bool
    answer: str

    # Multi-path loop results
    sub_results: list[SubQueryState]

    # AssemblerAgent
    citations: list[CitationState]
    sources: list[str]
    chunk_types: list[str]
    retrieval_scores: list[float]
    num_chunks: int
    partial: bool
    errors: list[str]

    # Controls / observability
    use_cache: bool
    top_k: int
    max_new_tokens: int
    temperature: float
    verbose: bool
    persist_session: bool
    timings: TimingState

    # Future nodes (optional placeholders)
    guardrails_ok: bool
    verified: bool


def empty_timings() -> TimingState:
    return {
        "rewrite_ms": 0.0,
        "decompose_ms": 0.0,
        "intent_ms": 0.0,
        "retrieve_ms": 0.0,
        "cache_ms": 0.0,
        "generate_ms": 0.0,
        "verify_ms": 0.0,
        "total_ms": 0.0,
    }


def initial_state(
    question: str,
    *,
    session_id: str = "",
    request_id: str | None = None,
    use_cache: bool = True,
    top_k: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    verbose: bool = False,
    persist_session: bool = True,
) -> GraphState:
    """Build the starting GraphState for one /ask turn."""
    return {
        "request_id": request_id or str(uuid.uuid4()),
        "session_id": session_id or "",
        "question": question,
        "retrieval_query": question,
        "rewrite": None,
        "decompose": None,
        "sub_queries": [],
        "is_multi": False,
        "intent": None,
        "hits": [],
        "dual_retrieval": False,
        "cache_hit": False,
        "answer": "",
        "sub_results": [],
        "citations": [],
        "sources": [],
        "chunk_types": [],
        "retrieval_scores": [],
        "num_chunks": 0,
        "partial": False,
        "errors": [],
        "use_cache": use_cache,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "verbose": verbose,
        "persist_session": persist_session,
        "timings": empty_timings(),
        "guardrails_ok": True,
        "verified": False,
    }


def merge_timing(state: GraphState, **deltas: float) -> TimingState:
    """Return updated timings dict (does not mutate state in place)."""
    current = dict(state.get("timings") or empty_timings())
    for key, value in deltas.items():
        current[key] = float(current.get(key, 0.0)) + float(value)
    return current  # type: ignore[return-value]


class Timer:
    """Simple stage timer: ``with Timer() as t: ...`` then ``t.ms``."""

    def __init__(self) -> None:
        self._start = 0.0
        self.ms = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000
