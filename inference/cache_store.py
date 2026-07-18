"""Redis Cloud answer cache for RAG generation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from inference.utils import load_dotenv

DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days
KEY_PREFIX = "onboarding:answer:"
# Bump when prompt, model, or corpus fingerprint semantics change.
CACHE_VERSION = os.environ.get("CACHE_VERSION", "v2")


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def chunk_fingerprint(results: list[dict]) -> str:
    ids: list[str] = []
    for hit in results:
        doc_id = hit.get("id")
        if not doc_id:
            meta = hit.get("metadata") or {}
            doc_id = str(meta.get("source", ""))
        ids.append(str(doc_id))
    return ",".join(ids)


def make_answer_key(
    query: str,
    results: list[dict],
    *,
    model_version: str = "",
    prompt_version: str = "",
) -> str:
    payload = "|".join(
        [
            CACHE_VERSION,
            model_version or "default",
            prompt_version or "default",
            normalize_query(query),
            chunk_fingerprint(results),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]
    return f"{KEY_PREFIX}{digest}"


def should_cache_answer(answer: str) -> bool:
    """Skip caching abstentions / empty replies."""
    if not answer or not answer.strip():
        return False
    lower = answer.lower()
    if "i don't have enough information" in lower:
        return False
    return True


class AnswerCache:
    """Thin Redis wrapper. No-ops if REDIS_URL missing or connection fails."""

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        ttl_seconds: int | None = None,
        enabled: bool = True,
        model_version: str = "",
        prompt_version: str = "system_v1",
    ):
        load_dotenv()
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds or int(
            os.environ.get("CACHE_TTL_SECONDS", DEFAULT_TTL_SECONDS)
        )
        self.redis_url = (redis_url or os.environ.get("REDIS_URL", "")).strip()
        self.model_version = model_version or os.environ.get("MODEL_VERSION", "lora_v2")
        self.prompt_version = prompt_version
        self._client = None
        self._connect_error: str | None = None

        if self.enabled and self.redis_url:
            self._connect()

    def _connect(self) -> None:
        try:
            import redis

            client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            client.ping()
            self._client = client
            self._connect_error = None
        except Exception as exc:
            self._client = None
            self._connect_error = str(exc)

    @property
    def available(self) -> bool:
        return self.enabled and self._client is not None

    def status_message(self) -> str:
        if not self.enabled:
            return "disabled"
        if not self.redis_url:
            return "REDIS_URL not set"
        if self._connect_error:
            return f"unavailable ({self._connect_error})"
        return "connected"

    def health(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "status": self.status_message(),
            "cache_version": CACHE_VERSION,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
        }

    def get(self, query: str, results: list[dict]) -> str | None:
        if not self.available:
            return None
        key = make_answer_key(
            query,
            results,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
        )
        try:
            raw = self._client.get(key)
            if not raw:
                return None
            data = json.loads(raw)
            return data.get("answer")
        except Exception:
            # Fail-open: treat as miss
            return None

    def set(self, query: str, results: list[dict], answer: str) -> bool:
        if not self.available or not should_cache_answer(answer):
            return False
        key = make_answer_key(
            query,
            results,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
        )
        payload = json.dumps(
            {
                "query": query,
                "answer": answer,
                "chunk_ids": chunk_fingerprint(results).split(","),
                "cache_version": CACHE_VERSION,
                "model_version": self.model_version,
                "prompt_version": self.prompt_version,
            }
        )
        try:
            self._client.setex(key, self.ttl_seconds, payload)
            return True
        except Exception:
            return False
