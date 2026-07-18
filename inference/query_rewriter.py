"""LLM-based query rewriting for follow-up questions."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from inference.session_store import Turn

DEFAULT_REWRITE_MODEL = "llama-3.1-8b-instant"
MAX_ASSISTANT_CHARS = 800
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


@dataclass
class RewriteResult:
    needs_rewrite: bool
    standalone_query: str
    topic_status: str
    entities: list[str]
    confidence: float
    reason: str = ""
    active_topic: str = ""
    topic_summary: str = ""


REWRITE_SYSTEM_PROMPT = """You manage conversation context for a FastAPI onboarding search system.

Your job:
1. Decide if the latest message (or any PART of it) depends on prior conversation
2. If yes, rewrite vague follow-up parts into standalone wording using active topic/entities
3. If the message ALSO asks about a different topic, keep that part intact — do not drop it
4. If the whole message is already standalone / a new topic, set needs_rewrite=false
5. Update active_topic and topic_summary for what the user is exploring now

Follow-up only (needs_rewrite=true → one standalone query):
- "methods?" after UploadFile → "What are the methods of fastapi.UploadFile?"
- "what about headers?" after HTTPException → "How is the headers parameter used in HTTPException?"

Mixed follow-up + other topic (needs_rewrite=true → ONE string that still contains BOTH parts):
- Prior topic UploadFile. Message: "attributes? and also what arguments does HTTPException take?"
  → "What are the attributes of fastapi.UploadFile and what arguments does fastapi.HTTPException take?"
- Prior topic UploadFile. Message: "attributes and methods?"
  → "What are the attributes and methods of fastapi.UploadFile?"

Do NOT rewrite (needs_rewrite=false):
- "How does OAuth2 work in FastAPI?"
- "What arguments does HTTPException take and how do I use UploadFile?" (already fully standalone)

Rules:
- Never answer the question
- Never invent APIs not in the history or the latest message
- Never drop a topic the user asked about
- Preserve exact technical names
- active_topic: short label; topic_summary: one sentence; entities: key APIs/concepts
- Output JSON only, no markdown
"""


def _load_dotenv() -> None:
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _format_turns(turns: list[Turn]) -> str:
    if not turns:
        return "(no prior conversation)"
    lines: list[str] = []
    for turn in turns:
        content = turn.content
        if turn.role == "assistant":
            content = _truncate(content, MAX_ASSISTANT_CHARS)
        lines.append(f"{turn.role.upper()}: {content}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _call_rewrite_llm(
    *,
    current_message: str,
    recent_turns: list[Turn],
    topic_summary: str,
    active_topic: str,
    entities: list[str],
    past_topics: list[str],
    model: str,
) -> dict:
    from groq import Groq

    past_topics_block = ""
    if past_topics:
        past_topics_block = (
            "\nPreviously discussed (do not carry into rewrite unless user returns to it):\n"
            + "\n".join(f"- {topic}" for topic in past_topics[-3:])
        )

    user_prompt = f"""Recent conversation:
{_format_turns(recent_turns)}

Active topic:
{active_topic or "(none yet)"}

Active topic summary:
{topic_summary or "(none yet)"}

Known entities:
{", ".join(entities) if entities else "(none yet)"}
{past_topics_block}

Latest user message:
{current_message}

Return JSON:
{{
  "needs_rewrite": true or false,
  "topic_status": "same" or "new" or "ambiguous",
  "standalone_query": "...",
  "active_topic": "...",
  "topic_summary": "one sentence about current subject",
  "entities": ["..."],
  "confidence": 0.0 to 1.0,
  "reason": "short explanation"
}}

NOTE: If only a FOLLOW-UP part is vague, rewrite that part using session context but KEEP any other topics in standalone_query.
NOTE: standalone_query must be searchable alone (or as a compound of standalone parts).
NOTE: topic_status must be one of: same, new, ambiguous
NOTE: Never drop a second topic the user mentioned.
"""

    client = Groq()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=350,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    return _extract_json(raw)


def _result_from_data(
    data: dict,
    *,
    current_message: str,
    entities: list[str],
) -> RewriteResult:
    topic_status = str(data.get("topic_status", "same"))
    standalone = str(data.get("standalone_query", current_message)).strip() or current_message
    result_entities = [str(x) for x in data.get("entities", [])]
    active_topic = str(data.get("active_topic", "")).strip()
    topic_summary = str(data.get("topic_summary", "")).strip()

    if topic_status == "new":
        return RewriteResult(
            needs_rewrite=False,
            standalone_query=current_message,
            topic_status="new",
            entities=result_entities,
            confidence=float(data.get("confidence", 0.9)),
            reason=str(data.get("reason", "new topic")),
            active_topic=active_topic,
            topic_summary=topic_summary,
        )

    needs_rewrite = _as_bool(data.get("needs_rewrite", False))
    if standalone.lower() != current_message.strip().lower():
        needs_rewrite = True

    return RewriteResult(
        needs_rewrite=needs_rewrite,
        standalone_query=standalone,
        topic_status=topic_status,
        entities=result_entities or list(entities),
        confidence=float(data.get("confidence", 0.0)),
        reason=str(data.get("reason", "")),
        active_topic=active_topic,
        topic_summary=topic_summary,
    )


def rewrite_query(
    *,
    current_message: str,
    recent_turns: list[Turn],
    topic_summary: str = "",
    active_topic: str = "",
    entities: list[str] | None = None,
    past_topics: list[str] | None = None,
    model: str | None = None,
) -> RewriteResult:
    """Rewrite a follow-up into a standalone retrieval query using a small LLM."""
    _load_dotenv()

    entities = entities or []
    past_topics = past_topics or []

    if not os.environ.get("GROQ_API_KEY"):
        return RewriteResult(
            needs_rewrite=False,
            standalone_query=current_message,
            topic_status="new" if not recent_turns else "same",
            entities=entities,
            confidence=0.0,
            reason="GROQ_API_KEY not set — add to .env",
        )

    model = model or os.environ.get("REWRITE_MODEL", DEFAULT_REWRITE_MODEL)

    try:
        data = _call_rewrite_llm(
            current_message=current_message,
            recent_turns=recent_turns,
            topic_summary=topic_summary,
            active_topic=active_topic,
            entities=entities,
            past_topics=past_topics,
            model=model,
        )
        return _result_from_data(data, current_message=current_message, entities=entities)
    except Exception as exc:
        return RewriteResult(
            needs_rewrite=False,
            standalone_query=current_message,
            topic_status="ambiguous",
            entities=entities,
            confidence=0.0,
            reason=f"rewrite failed: {exc}",
        )
