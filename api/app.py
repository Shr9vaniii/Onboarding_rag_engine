"""FastAPI application for the Onboarding RAG demo."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.rag_engine import MAX_QUESTION_CHARS, OnboardingRAGEngine
from inference.session_store import SessionStore
from inference.utils import load_dotenv
from inference.agents.graph import ask_graph
from inference.cache_store import AnswerCache

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"

_engine: OnboardingRAGEngine | None = None
_store: SessionStore | None = None
_engine_error: str | None = None
_cache: AnswerCache | None = None


def get_cache() -> AnswerCache:
    global _cache
    if _cache is None:
        load_dotenv()
        _cache = AnswerCache(enabled=os.environ.get("USE_CACHE", "1") != "0")
    return _cache



def get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store


def get_engine() -> OnboardingRAGEngine:
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    if _engine_error:
        raise RuntimeError(_engine_error)
    load_dotenv()
    backend = os.environ.get("RAG_BACKEND", "auto")
    try:
        _engine = OnboardingRAGEngine(
            backend=backend,
            inference_url=os.environ.get("INFERENCE_URL") or None,
            load_model=backend == "local",
            session_store=get_store(),
            use_cache=os.environ.get("USE_CACHE", "1") != "0",
        )
        return _engine
    except Exception as exc:
        _engine_error = str(exc)
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    try:
        get_engine()
        logger.info("RAG engine initialized (backend=%s)", _engine.backend if _engine else "?")
    except Exception as exc:
        logger.warning("RAG engine deferred init failed: %s", exc)
    yield


app = FastAPI(
    title="Onboarding RAG API",
    description="Conversational hybrid RAG for FastAPI onboarding Q&A",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class CreateSessionResponse(BaseModel):
    session_id: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str | None = None
    bypass_cache: bool = False
    top_k: int = Field(default=5, ge=1, le=10)
    verbose: bool = False


class AskResponse(BaseModel):
    request_id: str
    session_id: str
    question: str
    answer: str
    citations: list[dict[str, Any]]
    intent: str
    retrieval_query: str
    rewrite: dict[str, Any] | None
    decompose: dict[str, Any] | None
    dual_retrieval: bool
    cache_hit: bool
    partial: bool
    errors: list[str]
    timings: dict[str, float]
    num_chunks: int
    model_status: str


def _rewrite_payload(rewrite) -> dict[str, Any] | None:
    if rewrite is None:
        return None
    return {
        "needs_rewrite": rewrite.needs_rewrite,
        "standalone_query": rewrite.standalone_query,
        "topic_status": rewrite.topic_status,
        "confidence": rewrite.confidence,
        "active_topic": rewrite.active_topic,
        "reason": rewrite.reason,
    }


def _decompose_payload(decompose) -> dict[str, Any] | None:
    if decompose is None:
        return None
    return {
        "is_multi": decompose.is_multi,
        "sub_queries": decompose.sub_queries,
        "confidence": decompose.confidence,
        "reason": decompose.reason,
    }

def _rewrite_from_state(rewrite: dict | None) -> dict[str, Any] | None:
    if not rewrite:
        return None
    return {
        "needs_rewrite": rewrite.get("needs_rewrite"),
        "standalone_query": rewrite.get("standalone_query"),
        "topic_status": rewrite.get("topic_status"),
        "confidence": rewrite.get("confidence"),
        "active_topic": rewrite.get("active_topic"),
        "reason": rewrite.get("reason"),
    }


def _decompose_from_state(decompose: dict | None) -> dict[str, Any] | None:
    if not decompose:
        return None
    return {
        "is_multi": decompose.get("is_multi"),
        "sub_queries": decompose.get("sub_queries") or [],
        "confidence": decompose.get("confidence"),
        "reason": decompose.get("reason"),
    }


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"message": "UI missing; use POST /ask"})
    return FileResponse(index_path)


@app.get("/health")
def health():
    payload: dict[str, Any] = {
        "status": "degraded",
        "retrieval": {"ok": False},
        "cache": {"available": False},
        "generation": {"ok": False},
    }
    try:
        engine = get_engine()
        payload = engine.health()
        payload["status"] = "ok" if payload.get("ready") else "degraded"
        if not payload["generation"].get("ok"):
            payload["message"] = "Model waking up or INFERENCE_URL not configured"
    except Exception as exc:
        payload["status"] = "starting"
        payload["message"] = str(exc)
        payload["generation"] = {
            "ok": False,
            "detail": "Model waking up or configuration incomplete",
        }
    return payload


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session():
    session_id = get_store().create_session()
    return CreateSessionResponse(session_id=session_id)


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request):
    store = get_store()
    session_id = body.session_id
    if session_id:
        if not store.session_exists(session_id):
            raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    else:
        session_id = store.create_session()
    
    try:
        engine = get_engine()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Generation backend not ready (model waking up or misconfigured).",
                "error": str(exc),
            },
        ) from exc

    request_id = request.headers.get("x-request-id") or None
    try:
        state = ask_graph(
            body.question.strip(),
            session_store=store,
            answer_cache=get_cache(),
            session_id=session_id,
            use_cache=not body.bypass_cache,
            top_k=body.top_k,
            request_id=request_id,
            verbose=body.verbose,
            inference_url=os.environ.get("INFERENCE_URL") or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("ask failed")
        raise HTTPException(
            status_code=502,
            detail={"message": "Upstream generation or retrieval failed.", "error": str(exc)},
        ) from exc
    intent = (state.get("intent") or {}).get("intent_name") or "UNKNOWN"
    timings = state.get("timings") or {}
    # round like before
    timings = {k: round(float(v), 2) for k, v in timings.items()}
    return AskResponse(
        request_id=state.get("request_id") or "",
        session_id=session_id,
        question=state.get("question") or body.question,
        answer=state.get("answer") or "",
        citations=list(state.get("citations") or []),
        intent=intent,
        retrieval_query=state.get("retrieval_query") or "",
        rewrite=_rewrite_from_state(state.get("rewrite")),
        decompose=_decompose_from_state(state.get("decompose")),
        dual_retrieval=bool(state.get("dual_retrieval")),
        cache_hit=bool(state.get("cache_hit")),
        partial=bool(state.get("partial")),
        errors=list(state.get("errors") or []),
        timings=timings,
        num_chunks=int(state.get("num_chunks") or 0),
        model_status="partial" if state.get("partial") else "ready",
    )


def create_app() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn

    load_dotenv()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api.app:app", host=host, port=port, reload=False)
