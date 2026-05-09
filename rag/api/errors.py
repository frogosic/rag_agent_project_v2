import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag.application.errors import (
    EmptyRetrievalError,
    GenerationError,
    InvalidFiltersError,
    RAGServiceError,
    RetrievalError,
)

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        logger.warning("Request validation failed for path=%s: %s", request.url.path, exc)

        return JSONResponse(
            status_code=422,
            content={
                "error": "request_validation_failed",
                "message": "Request validation failed.",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(InvalidFiltersError)
    async def invalid_filters_handler(
        request: Request,
        exc: InvalidFiltersError,
    ) -> JSONResponse:
        logger.warning("Invalid filters for path=%s: %s", request.url.path, exc)

        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_filters",
                "message": str(exc),
            },
        )

    @app.exception_handler(EmptyRetrievalError)
    async def empty_retrieval_handler(
        request: Request,
        exc: EmptyRetrievalError,
    ) -> JSONResponse:
        logger.info("Empty retrieval for path=%s: %s", request.url.path, exc)

        return JSONResponse(
            status_code=200,
            content={
                "error": "empty_retrieval",
                "message": str(exc),
                "answer": "I do not have enough retrieved context to answer this question reliably.",
                "retrieved_chunks": [],
            },
        )

    @app.exception_handler(RetrievalError)
    async def retrieval_error_handler(
        request: Request,
        exc: RetrievalError,
    ) -> JSONResponse:
        logger.exception("Retrieval failure for path=%s", request.url.path)

        return JSONResponse(
            status_code=503,
            content={
                "error": "retrieval_unavailable",
                "message": "Retrieval service is unavailable or misconfigured.",
            },
        )

    @app.exception_handler(GenerationError)
    async def generation_error_handler(
        request: Request,
        exc: GenerationError,
    ) -> JSONResponse:
        logger.exception("Generation failure for path=%s", request.url.path)

        return JSONResponse(
            status_code=502,
            content={
                "error": "generation_failed",
                "message": "Answer generation failed.",
            },
        )

    @app.exception_handler(RAGServiceError)
    async def rag_service_error_handler(
        request: Request,
        exc: RAGServiceError,
    ) -> JSONResponse:
        logger.exception("RAG service failure for path=%s", request.url.path)

        return JSONResponse(
            status_code=500,
            content={
                "error": "rag_service_error",
                "message": "The RAG service failed unexpectedly.",
            },
        )
