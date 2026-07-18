"""
Onboarding RAG engine — rewrite → decompose → retrieve → cache → generate.

Usage:
  python -m inference.rag_engine "How do I use Depends with a database session?"
  python -m inference.rag_engine "..." --retrieve-only
  python -m inference.rag_engine "..." --backend remote --inference-url https://xxxx.ngrok-free.app
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.cache_store import AnswerCache
from inference.context import SYSTEM_PROMPT, format_multi_chunk_user_message
from inference.intent_classifier import ClassifiedQuery, classify_intent
from inference.query_decomposer import (
    DecomposeResult,
    SubQueryAnswer,
    decompose_query,
)
from inference.query_rewriter import RewriteResult, rewrite_query
from inference.retriever import (
    retrieve_dual,
    retrieve_for_query,
    retrieve_multi,
    should_dual_retrieve,
)
from inference.session_store import SessionStore
from inference.utils import extract_assistant_reply, load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_DUAL_CONFIDENCE_THRESHOLD = 0.75
MAX_QUESTION_CHARS = 2000
REMOTE_MAX_RETRIES = 3
REMOTE_BACKOFF_SECONDS = 1.5
EMPTY_ANSWER = (
    "**Direct Answer:**\n"
    "I don't have enough information in the provided context to answer this accurately.\n\n"
    "**What I found:**\n"
    "No relevant documents were retrieved from the knowledge base.\n\n"
    "**Recommendation:**\n"
    "Try rephrasing the question or ask your team lead."
)
PARTIAL_FAILURE_PREFIX = (
    "**Note:** Part of this multi-part question could not be answered "
    "because generation failed for one or more sub-queries.\n\n"
)

DEFAULT_BASE_MODEL = "unsloth/Llama-3.1-8B-Instruct"
DEFAULT_LORA_PATH = ROOT / "model" / "onboarding_lora_v2"


def _lora_weights_exist(lora_path: Path) -> bool:
    return any(
        (lora_path / name).exists()
        for name in (
            "adapter_model.safetensors",
            "adapter_model.bin",
            "model.safetensors",
        )
    )


def _generate_remote(
    messages: list[dict],
    *,
    url: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: int = 120,
    api_key: str | None = None,
    max_retries: int = REMOTE_MAX_RETRIES,
) -> str:
    base = url.rstrip("/")
    payload = json.dumps(
        {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    key = api_key or os.environ.get("INFERENCE_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            f"{base}/generate",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            answer = data.get("answer", "")
            if not answer:
                raise RuntimeError(f"Inference server returned empty answer: {data}")
            return answer
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            detail = body
            try:
                err_json = json.loads(body)
                detail = err_json.get("detail", body)
            except json.JSONDecodeError:
                pass
            last_error = RuntimeError(f"Inference server error {exc.code}: {detail}")
            # Retry only on transient server/network-ish codes
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = RuntimeError(
                f"Cannot reach inference server at {base}. "
                "Check INFERENCE_URL and that Colab/ngrok/RunPod is running."
            )
            last_error.__cause__ = exc
        except Exception as exc:
            last_error = exc

        if attempt < max_retries:
            sleep_for = REMOTE_BACKOFF_SECONDS * attempt
            logger.warning(
                "Remote generation attempt %s/%s failed (%s); retrying in %.1fs",
                attempt,
                max_retries,
                last_error,
                sleep_for,
            )
            time.sleep(sleep_for)

    assert last_error is not None
    raise last_error


@dataclass
class StageTimings:
    rewrite_ms: float = 0.0
    decompose_ms: float = 0.0
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    cache_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "rewrite_ms": round(self.rewrite_ms, 2),
            "decompose_ms": round(self.decompose_ms, 2),
            "retrieve_ms": round(self.retrieve_ms, 2),
            "generate_ms": round(self.generate_ms, 2),
            "cache_ms": round(self.cache_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


@dataclass
class RAGResponse:
    question: str
    answer: str
    classification: ClassifiedQuery
    sources: list[str]
    chunk_types: list[str]
    retrieval_scores: list[float]
    num_chunks: int
    retrieval_query: str = ""
    rewrite: RewriteResult | None = None
    session_id: str = ""
    dual_retrieval: bool = False
    decompose: DecomposeResult | None = None
    sub_answers: list[SubQueryAnswer] | None = None
    cache_hit: bool = False
    request_id: str = ""
    timings: StageTimings = field(default_factory=StageTimings)
    partial: bool = False
    errors: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)

    @property
    def is_multi(self) -> bool:
        return bool(self.decompose and self.decompose.is_multi)

    @property
    def source(self) -> str:
        return self.sources[0] if self.sources else ""

    @property
    def chunk_type(self) -> str:
        return self.chunk_types[0] if self.chunk_types else ""

    @property
    def retrieval_score(self) -> float:
        return self.retrieval_scores[0] if self.retrieval_scores else 0.0


class OnboardingRAGEngine:
    def __init__(
        self,
        *,
        lora_path: str | Path | None = None,
        base_model: str = DEFAULT_BASE_MODEL,
        max_seq_length: int = 2048,
        backend: str = "auto",
        inference_url: str | None = None,
        load_model: bool = True,
        session_store: SessionStore | None = None,
        answer_cache: AnswerCache | None = None,
        use_cache: bool = True,
    ):
        load_dotenv()
        self.lora_path = Path(lora_path) if lora_path else DEFAULT_LORA_PATH
        self.base_model = base_model
        self.max_seq_length = max_seq_length
        self.inference_url = (inference_url or os.environ.get("INFERENCE_URL", "")).rstrip("/")
        self.backend = self._resolve_backend(backend)
        self.model = None
        self.tokenizer = None
        self.session_store = session_store or SessionStore()
        self.default_use_cache = use_cache
        self.answer_cache = (
            answer_cache if answer_cache is not None else AnswerCache(enabled=use_cache)
        )
        if load_model and self.backend == "local":
            self._load_model()

    def _resolve_backend(self, backend: str) -> str:
        if backend not in ("auto", "local", "remote"):
            raise ValueError(f"Unknown backend: {backend}")
        if backend == "remote":
            if not self.inference_url:
                raise ValueError(
                    "Remote backend requires --inference-url or INFERENCE_URL in .env"
                )
            return "remote"
        if backend == "local":
            return "local"
        if self.inference_url:
            logger.info("Using remote inference at %s", self.inference_url)
            return "remote"
        try:
            import torch

            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        try:
            import unsloth  # noqa: F401

            has_unsloth = True
        except ImportError:
            has_unsloth = False
        if has_cuda and has_unsloth and _lora_weights_exist(self.lora_path):
            return "local"
        raise RuntimeError(
            "No GPU/LoRA available locally. Run deploy/colab/inference_server.ipynb "
            "or deploy/runpod, then set INFERENCE_URL, or pass --backend remote."
        )

    def _load_model(self) -> None:
        if not _lora_weights_exist(self.lora_path):
            raise FileNotFoundError(
                f"Missing adapter weights in {self.lora_path}. "
                "Copy adapter_model.safetensors from Google Drive / RunPod volume."
            )
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"Loading base model (transformers 4bit): {self.base_model}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        print(f"Attaching LoRA from: {self.lora_path}")
        self.model = PeftModel.from_pretrained(self.model, str(self.lora_path))
        self.model.eval()
        print("Model ready.")

    def health(self) -> dict:
        retrieval_ok = False
        retrieval_error = ""
        try:
            from retrieval.hybrid_retrieval import get_collection

            count = get_collection().count()
            retrieval_ok = count > 0
            if not retrieval_ok:
                retrieval_error = "Chroma collection is empty"
        except Exception as exc:
            retrieval_error = str(exc)

        generation_ok = False
        generation_detail = self.backend
        if self.backend == "remote":
            if self.inference_url:
                generation_ok = True
                generation_detail = self.inference_url
            else:
                generation_detail = "INFERENCE_URL missing"
        elif self.backend == "local":
            generation_ok = self.model is not None

        cache = self.answer_cache.health() if self.answer_cache else {"available": False}
        ready = retrieval_ok and generation_ok
        return {
            "ready": ready,
            "retrieval": {"ok": retrieval_ok, "error": retrieval_error},
            "cache": cache,
            "generation": {
                "ok": generation_ok,
                "backend": self.backend,
                "detail": generation_detail,
            },
        }

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        verbose: bool = False,
    ) -> tuple[ClassifiedQuery, list[dict]]:
        return retrieve_for_query(question, n_final=top_k, verbose=verbose)

    def _apply_rewrite_to_session(
        self,
        session_id: str,
        session,
        rewrite: RewriteResult,
    ) -> None:
        if rewrite.topic_status == "new":
            past_topics = list(session.past_topics)
            if session.active_topic:
                past_topics.append(session.active_topic)
            self.session_store.update_session_context(
                session_id,
                active_topic=rewrite.active_topic,
                topic_summary=rewrite.topic_summary,
                entities=rewrite.entities,
                past_topics=past_topics,
            )
            return

        self.session_store.update_session_context(
            session_id,
            active_topic=rewrite.active_topic or session.active_topic,
            topic_summary=rewrite.topic_summary or session.topic_summary,
            entities=rewrite.entities or session.entities,
        )

    def _prepare_retrieval_query(
        self,
        question: str,
        *,
        session_id: str | None = None,
        verbose: bool = False,
        timings: StageTimings | None = None,
    ) -> tuple[str, RewriteResult | None]:
        if not session_id:
            return question, None

        session = self.session_store.get_session(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")

        recent_turns = self.session_store.get_recent_exchanges(session_id, max_exchanges=3)
        t0 = time.perf_counter()
        rewrite = rewrite_query(
            current_message=question,
            recent_turns=recent_turns,
            topic_summary=session.topic_summary,
            active_topic=session.active_topic,
            entities=session.entities,
            past_topics=session.past_topics,
        )
        if timings is not None:
            timings.rewrite_ms = (time.perf_counter() - t0) * 1000

        self._apply_rewrite_to_session(session_id, session, rewrite)

        retrieval_query = (
            rewrite.standalone_query if rewrite.needs_rewrite else question
        )
        if verbose:
            if rewrite.needs_rewrite:
                print(f"Rewrite:   {question!r}")
                print(f"Search as: {retrieval_query!r}")
            else:
                print(f"Rewrite:   skipped ({rewrite.reason or 'not a follow-up'})")
            print(
                f"Topic:     {rewrite.topic_status} "
                f"(confidence={rewrite.confidence:.2f})"
            )
        return retrieval_query, rewrite

    def retrieve_with_rewrite(
        self,
        question: str,
        retrieval_query: str,
        rewrite: RewriteResult | None,
        *,
        top_k: int = DEFAULT_TOP_K,
        verbose: bool = False,
        dual_threshold: float = DEFAULT_DUAL_CONFIDENCE_THRESHOLD,
    ) -> tuple[ClassifiedQuery, list[dict], bool]:
        use_dual = (
            rewrite is not None
            and should_dual_retrieve(
                needs_rewrite=rewrite.needs_rewrite,
                confidence=rewrite.confidence,
                topic_status=rewrite.topic_status,
                original=question,
                retrieval_query=retrieval_query,
                threshold=dual_threshold,
            )
        )
        if use_dual:
            classified, results = retrieve_dual(
                question,
                retrieval_query,
                n_final=top_k,
                verbose=verbose,
                confidence_threshold=dual_threshold,
            )
            return classified, results, True
        classified, results = self.retrieve(retrieval_query, top_k=top_k, verbose=verbose)
        return classified, results, False

    def _generate_from_messages(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        if self.backend == "remote":
            return _generate_remote(
                messages,
                url=self.inference_url,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Local model not loaded")
        import torch

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_cache=True,
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return extract_assistant_reply(decoded)

    def _answer_one(
        self,
        question: str,
        results: list[dict],
        *,
        max_new_tokens: int,
        temperature: float,
        use_cache: bool,
        verbose: bool = False,
        timings: StageTimings | None = None,
    ) -> tuple[str, bool]:
        """Return (answer, cache_hit). Never mutates engine-wide cache flags."""
        if not results:
            return EMPTY_ANSWER, False

        if use_cache and self.answer_cache is not None:
            t0 = time.perf_counter()
            cached = self.answer_cache.get(question, results)
            if timings is not None:
                timings.cache_ms += (time.perf_counter() - t0) * 1000
            if cached:
                if verbose:
                    print(f"Cache:     HIT for {question!r}")
                return cached, True
            if verbose:
                print(
                    f"Cache:     MISS ({self.answer_cache.status_message()}) for {question!r}"
                )

        user_content = format_multi_chunk_user_message(question, results)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        t1 = time.perf_counter()
        answer = self._generate_from_messages(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        if timings is not None:
            timings.generate_ms += (time.perf_counter() - t1) * 1000

        if use_cache and self.answer_cache is not None:
            t2 = time.perf_counter()
            self.answer_cache.set(question, results, answer)
            if timings is not None:
                timings.cache_ms += (time.perf_counter() - t2) * 1000
        return answer, False

    @staticmethod
    def _pack_hits(results: list[dict]) -> tuple[list[str], list[str], list[float], list[dict]]:
        sources = [str(h["metadata"].get("source", "")) for h in results]
        chunk_types = [h.get("chunk_type", "") for h in results]
        scores = [
            float(h.get("final_score", h.get("rerank_score", h.get("rrf_score", 0))))
            for h in results
        ]
        citations = []
        for h in results:
            meta = h.get("metadata") or {}
            citations.append(
                {
                    "id": h.get("id", ""),
                    "source": str(meta.get("source", "")),
                    "name": str(meta.get("name") or meta.get("qualified_name", "")),
                    "chunk_type": h.get("chunk_type", ""),
                    "score": float(
                        h.get("final_score", h.get("rerank_score", h.get("rrf_score", 0)))
                    ),
                    "snippet": (h.get("content") or "")[:240],
                }
            )
        return sources, chunk_types, scores, citations

    @staticmethod
    def _format_multi_answer(sub_answers: list[SubQueryAnswer], *, partial: bool) -> str:
        parts: list[str] = []
        if partial:
            parts.append(PARTIAL_FAILURE_PREFIX.strip())
        for i, sa in enumerate(sub_answers, 1):
            parts.append(f"### Question {i}: {sa.query}\n\n{sa.answer}")
        return "\n\n---\n\n".join(parts)

    def generate(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        verbose: bool = False,
        session_id: str | None = None,
        use_cache: bool | None = None,
        request_id: str | None = None,
        persist_session: bool = True,
    ) -> RAGResponse:
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(
                f"Question exceeds {MAX_QUESTION_CHARS} character limit "
                f"(got {len(question)})."
            )

        cache_enabled = self.default_use_cache if use_cache is None else use_cache
        req_id = request_id or str(uuid.uuid4())
        timings = StageTimings()
        started = time.perf_counter()
        logger.info("request_id=%s session_id=%s start", req_id, session_id or "-")

        retrieval_query, rewrite = self._prepare_retrieval_query(
            question,
            session_id=session_id,
            verbose=verbose,
            timings=timings,
        )

        t_decomp = time.perf_counter()
        decompose = decompose_query(retrieval_query)
        timings.decompose_ms = (time.perf_counter() - t_decomp) * 1000

        if verbose:
            print(f"Cache:     {self.answer_cache.status_message()}")
            if decompose.is_multi:
                print(f"Multi:     yes ({len(decompose.sub_queries)} sub-queries)")
            else:
                print(f"Multi:     no ({decompose.reason or 'single question'})")

        # --- Multi-query path ---
        if decompose.is_multi:
            t_ret = time.perf_counter()
            multi = retrieve_multi(
                decompose.sub_queries,
                n_final=top_k,
                verbose=verbose,
            )
            timings.retrieve_ms = (time.perf_counter() - t_ret) * 1000

            sub_answers: list[SubQueryAnswer] = []
            all_sources: list[str] = []
            all_types: list[str] = []
            all_scores: list[float] = []
            all_citations: list[dict] = []
            any_cache_hit = False
            errors: list[str] = []
            partial = False

            for (classified_i, results_i), sq in zip(multi, decompose.sub_queries):
                try:
                    answer_i, hit_i = self._answer_one(
                        sq,
                        results_i,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        use_cache=cache_enabled,
                        verbose=verbose,
                        timings=timings,
                    )
                except Exception as exc:
                    partial = True
                    err = f"Sub-query failed ({sq}): {exc}"
                    errors.append(err)
                    logger.exception("request_id=%s %s", req_id, err)
                    answer_i, hit_i = (
                        (
                            "**Direct Answer:**\n"
                            "I couldn't generate an answer for this part due to a "
                            "temporary generation error. Other parts below may still "
                            "be useful.\n\n"
                            f"**Error:** {exc}"
                        ),
                        False,
                    )
                any_cache_hit = any_cache_hit or hit_i
                sources_i, types_i, scores_i, cites_i = self._pack_hits(results_i)
                sub_answers.append(
                    SubQueryAnswer(
                        query=sq,
                        answer=answer_i,
                        sources=sources_i,
                        chunk_types=types_i,
                        retrieval_scores=scores_i,
                        num_chunks=len(results_i),
                    )
                )
                all_sources.extend(sources_i)
                all_types.extend(types_i)
                all_scores.extend(scores_i)
                all_citations.extend(cites_i)

            primary = multi[0][0] if multi else classify_intent(question)
            timings.total_ms = (time.perf_counter() - started) * 1000
            response = RAGResponse(
                question=question,
                answer=self._format_multi_answer(sub_answers, partial=partial),
                classification=primary,
                sources=all_sources,
                chunk_types=all_types,
                retrieval_scores=all_scores,
                num_chunks=sum(sa.num_chunks for sa in sub_answers),
                retrieval_query=retrieval_query,
                rewrite=rewrite,
                session_id=session_id or "",
                dual_retrieval=False,
                decompose=decompose,
                sub_answers=sub_answers,
                cache_hit=any_cache_hit,
                request_id=req_id,
                timings=timings,
                partial=partial,
                errors=errors,
                citations=all_citations,
            )
            if persist_session and session_id:
                self.session_store.add_turn(session_id, question, response.answer)
            logger.info(
                "request_id=%s multi done total_ms=%.1f partial=%s",
                req_id,
                timings.total_ms,
                partial,
            )
            return response

        # --- Single-query path ---
        t_ret = time.perf_counter()
        classified, results, dual_used = self.retrieve_with_rewrite(
            question,
            retrieval_query,
            rewrite,
            top_k=top_k,
            verbose=verbose,
        )
        timings.retrieve_ms = (time.perf_counter() - t_ret) * 1000
        sources, chunk_types, scores, citations = self._pack_hits(results)

        if not results:
            timings.total_ms = (time.perf_counter() - started) * 1000
            response = RAGResponse(
                question=question,
                answer=EMPTY_ANSWER,
                classification=classified,
                sources=[],
                chunk_types=[],
                retrieval_scores=[],
                num_chunks=0,
                retrieval_query=retrieval_query,
                rewrite=rewrite,
                session_id=session_id or "",
                dual_retrieval=dual_used,
                decompose=decompose,
                cache_hit=False,
                request_id=req_id,
                timings=timings,
                citations=[],
            )
            if persist_session and session_id:
                self.session_store.add_turn(session_id, question, response.answer)
            return response

        answer_question = (
            retrieval_query
            if rewrite is not None and rewrite.needs_rewrite
            else question
        )
        answer, cache_hit = self._answer_one(
            answer_question,
            results,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_cache=cache_enabled,
            verbose=verbose,
            timings=timings,
        )
        timings.total_ms = (time.perf_counter() - started) * 1000
        response = RAGResponse(
            question=question,
            answer=answer,
            classification=classified,
            sources=sources,
            chunk_types=chunk_types,
            retrieval_scores=scores,
            num_chunks=len(results),
            retrieval_query=retrieval_query,
            rewrite=rewrite,
            session_id=session_id or "",
            dual_retrieval=dual_used,
            decompose=decompose,
            cache_hit=cache_hit,
            request_id=req_id,
            timings=timings,
            citations=citations,
        )
        if persist_session and session_id:
            self.session_store.add_turn(session_id, question, response.answer)
        logger.info(
            "request_id=%s done total_ms=%.1f cache_hit=%s dual=%s",
            req_id,
            timings.total_ms,
            cache_hit,
            dual_used,
        )
        return response


def main() -> None:
    # Windows consoles default to cp1252, which can't print corpus chars like '→'
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Onboarding RAG assistant")
    parser.add_argument("question", help="Engineer question")
    parser.add_argument(
        "--lora-path",
        default=os.environ.get("LORA_PATH", str(DEFAULT_LORA_PATH)),
        help="Path to fine-tuned LoRA adapter directory",
    )
    parser.add_argument(
        "--base-model",
        default=os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL),
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "local", "remote"),
        default="auto",
        help="auto: remote if INFERENCE_URL set, else local GPU",
    )
    parser.add_argument(
        "--inference-url",
        default=os.environ.get("INFERENCE_URL", ""),
        help="Colab/ngrok or RunPod URL",
    )
    parser.add_argument("--retrieve-only", action="store_true", help="Skip generation")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--session-id", default="")
    parser.add_argument("--new-session", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    store = SessionStore()
    session_id = args.session_id.strip()
    if args.new_session:
        session_id = store.create_session()
        print(f"Session:   {session_id}")
    elif session_id and not store.session_exists(session_id):
        raise SystemExit(f"Unknown session_id: {session_id}")

    if args.retrieve_only:
        classified, results = retrieve_for_query(
            args.question, n_final=args.top_k, verbose=True
        )
        if not results:
            print("\nNo chunks retrieved.")
            return
        from inference.context import build_training_context

        print(f"\n--- Top {len(results)} chunks ---")
        for i, hit in enumerate(results, 1):
            meta = hit["metadata"]
            name = meta.get("name") or meta.get("qualified_name", "")
            print(
                f"\n[{i}] final={hit.get('final_score', 0):.3f} "
                f"rerank={hit.get('rerank_score', 0):.3f} "
                f"boost={hit.get('entity_boost', 0):+.1f} | {name}"
            )
            print(
                build_training_context(
                    hit["content"],
                    meta,
                    hit["chunk_type"],
                )[:800]
            )
        return

    engine = OnboardingRAGEngine(
        lora_path=args.lora_path,
        base_model=args.base_model,
        backend=args.backend,
        inference_url=args.inference_url,
        session_store=store,
        use_cache=not args.no_cache,
    )
    response = engine.generate(
        args.question,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        verbose=args.verbose,
        session_id=session_id or None,
        use_cache=not args.no_cache,
    )

    if args.verbose:
        print(f"\nRequest:    {response.request_id}")
        print(f"Backend:    {engine.backend}")
        print(f"Cache:      {'HIT' if response.cache_hit else 'MISS'}")
        print(f"Timings:    {response.timings.as_dict()}")
        if response.dual_retrieval:
            print("Retrieval:  dual (original + rewritten)")
        if response.partial:
            print(f"Partial:    {response.errors}")

    print("\n" + response.answer)


if __name__ == "__main__":
    main()
