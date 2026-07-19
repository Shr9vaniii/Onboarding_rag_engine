"""Conversation Agent."""

from __future__ import annotations

from inference.agents.state import GraphState,RewriteState,Timer,merge_timing
from inference.query_rewriter import RewriteResult, rewrite_query
from inference.session_store import SessionStore

def _rewrite_to_state(rewrite:RewriteResult)-> RewriteState:
    return {
        "needs_rewrite": rewrite.needs_rewrite,
        "standalone_query": rewrite.standalone_query,
        "topic_status": rewrite.topic_status,
        "entities": list(rewrite.entities),
        "confidence": rewrite.confidence,
        "reason": rewrite.reason,
        "active_topic": rewrite.active_topic,
        "topic_summary": rewrite.topic_summary,
    }

def _apply_rewrite_to_session(
    session_store:SessionStore,
    session_id:str,
    session,
    rewrite:RewriteResult,
)->None:
    if rewrite.topic_status == "new":
            past_topics = list(session.past_topics)
            if session.active_topic:
                past_topics.append(session.active_topic)
            session_store.update_session_context(
                session_id,
                active_topic=rewrite.active_topic,
                topic_summary=rewrite.topic_summary,
                entities=rewrite.entities,
                past_topics=past_topics,
            )
            return

    session_store.update_session_context(
            session_id,
            active_topic=rewrite.active_topic or session.active_topic,
            topic_summary=rewrite.topic_summary or session.topic_summary,
            entities=rewrite.entities or session.entities,
        )

def run_conversation_agent(
    state:GraphState,
    *,
    session_store:SessionStore,
)->dict:
    question = state["question"]
    session_id = (state.get("session_id") or "").strip()
    if not session_id:
        return {
            "retrieval_query": question,
            "rewrite": None,
        }
    session = session_store.get_session(session_id)
    if session is None:
        raise KeyError(f"Unknown session_id: {session_id}")
    recent_turns = session_store.get_recent_exchanges(session_id, max_exchanges=3)
    with Timer() as t:
        rewrite = rewrite_query(
            current_message=question,
            recent_turns=recent_turns,
            topic_summary=session.topic_summary,
            active_topic=session.active_topic,
            entities=session.entities,
            past_topics=session.past_topics,
        )
    _apply_rewrite_to_session(session_store, session_id, session, rewrite)
    retrieval_query = (
        rewrite.standalone_query if rewrite.needs_rewrite else question
    )
    return {
        "retrieval_query": retrieval_query,
        "rewrite": _rewrite_to_state(rewrite),
        "timings": merge_timing(state, rewrite_ms=t.ms),
    }