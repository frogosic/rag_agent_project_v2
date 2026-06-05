import logging
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from typing import Any

from rag.api.schemas import AskRequest, AskResponse
from rag.application.rag_service import RAGService
from rag.application.run_store import RunStore
from rag.api.errors import register_exception_handlers

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Agent Project v2",
    version="0.1.0",
)

register_exception_handlers(app)


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Create and cache the RAG service for API usage."""
    return RAGService()


@lru_cache(maxsize=1)
def get_run_store() -> RunStore:
    """Create and cache the run store for API usage."""
    return RunStore()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    rag_service: RAGService = get_rag_service()
    return rag_service.readiness_check()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict[str, Any]:
    logger.info("Received ask request: %s", request.query)

    rag_service: RAGService = get_rag_service()
    run_store: RunStore = get_run_store()

    result = rag_service.answer(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
        retrieval_mode=request.retrieval_mode,
    )

    output_path: Path = run_store.write_ask_run(result)
    logger.info("Persisted ask run %s to %s", result["run_id"], output_path)

    return result
