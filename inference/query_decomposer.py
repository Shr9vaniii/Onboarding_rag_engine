"""LLM-based decomposition of compound user questions into search sub-queries."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DECOMPOSE_MODEL = "llama-3.1-8b-instant"
MAX_SUB_QUERIES = 3
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

DECOMPOSE_SYSTEM_PROMPT = """You decompose user questions for a FastAPI onboarding search system.

Your job:
1. Decide if the message contains multiple searchable parts (different topics OR different aspects of one topic)
2. If yes, split into separate STANDALONE search queries (max 3)
3. If no, return a single query (the message itself)

CRITICAL — every sub_query must be self-contained:
- A sub_query must make sense ALONE, with no prior clause
- Carry over the shared topic/API/entity into EVERY sub_query
- Resolve vague phrases like "what arguments", "how", "required", "parameters", "it", "that"
  using the topic from the rest of the message

Bad split (DO NOT do this):
Input: "How to upload a file and what arguments are required?"
→ ["How to upload a file", "What arguments are required"]  ← second is useless alone

Good split:
Input: "How to upload a file and what arguments are required?"
→ ["How do I upload a file in FastAPI?", "What arguments/parameters are required for file upload in FastAPI?"]

More good examples (is_multi=true):
- "What args does HTTPException take and how do I use UploadFile?"
  → ["What arguments does fastapi.HTTPException take?", "How do I use fastapi.UploadFile?"]
- "Explain Depends and also how OAuth2 works"
  → ["How does FastAPI Depends work?", "How does OAuth2 work in FastAPI?"]
- "How do I use UploadFile and what is its signature?"
  → ["How do I use fastapi.UploadFile?", "What is the signature/arguments of fastapi.UploadFile?"]

is_multi=false examples:
- "What arguments does HTTPException take?"
- "How do I upload files with UploadFile in FastAPI?"
- "Tell me about HTTPException headers"  ← one topic, do not split

Rules:
- Never answer the questions
- Never invent APIs not implied by the message
- Preserve exact technical names when present
- Output JSON only
"""


@dataclass
class DecomposeResult:
    is_multi: bool
    sub_queries: list[str]
    confidence: float
    reason: str = ""


@dataclass
class SubQueryAnswer:
    query: str
    answer: str
    sources: list[str] = field(default_factory=list)
    chunk_types: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    num_chunks: int = 0


def _load_dotenv() -> None:
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


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


def _normalize_sub_queries(raw: object, fallback: str) -> list[str]:
    if not isinstance(raw, list):
        return [fallback]
    queries: list[str] = []
    for item in raw:
        q = str(item).strip()
        if q and q not in queries:
            queries.append(q)
        if len(queries) >= MAX_SUB_QUERIES:
            break
    return queries or [fallback]


def decompose_query(
    query: str,
    *,
    model: str | None = None,
) -> DecomposeResult:
    """Split a compound question into up to 3 self-contained search sub-queries."""
    _load_dotenv()
    query = query.strip()
    if not query:
        return DecomposeResult(
            is_multi=False,
            sub_queries=[""],
            confidence=0.0,
            reason="empty query",
        )

    if not os.environ.get("GROQ_API_KEY"):
        return DecomposeResult(
            is_multi=False,
            sub_queries=[query],
            confidence=0.0,
            reason="GROQ_API_KEY not set — add to .env",
        )

    from groq import Groq

    model = model or os.environ.get("DECOMPOSE_MODEL", DEFAULT_DECOMPOSE_MODEL)
    user_prompt = f"""User message:
{query}

Return JSON:
{{
  "is_multi": true or false,
  "sub_queries": ["..."],
  "confidence": 0.0 to 1.0,
  "reason": "short explanation"
}}

If is_multi is false, sub_queries has exactly one string.
If is_multi is true, sub_queries has 2–{MAX_SUB_QUERIES} strings.

Each sub_query MUST be a complete standalone search question that includes the shared topic
(e.g. file upload, HTTPException). Never leave a vague fragment like "what arguments are required".
"""

    try:
        client = Groq()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=350,
            response_format={"type": "json_object"},
        )
        data = _extract_json(response.choices[0].message.content or "")
        sub_queries = _normalize_sub_queries(data.get("sub_queries"), query)
        is_multi = _as_bool(data.get("is_multi", False)) and len(sub_queries) > 1
        if not is_multi:
            sub_queries = [sub_queries[0] if sub_queries else query]

        return DecomposeResult(
            is_multi=is_multi,
            sub_queries=sub_queries,
            confidence=float(data.get("confidence", 0.0)),
            reason=str(data.get("reason", "")),
        )
    except Exception as exc:
        return DecomposeResult(
            is_multi=False,
            sub_queries=[query],
            confidence=0.0,
            reason=f"decompose failed: {exc}",
        )
