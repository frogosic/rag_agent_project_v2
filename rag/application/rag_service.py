import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from rag.application.errors import (
    EmptyRetrievalError,
    GenerationError,
    InvalidFiltersError,
    RetrievalError,
)
from rag.generation.answer_service import AnswerService
from rag.indexing.embedding_service import EmbeddingService
from rag.stores.qdrant_store import QdrantStore
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.modes import DEFAULT_RETRIEVAL_MODE, RetrievalMode

logger = logging.getLogger(__name__)


class RAGService:
    """Application-level service for retrieval-augmented answering."""

    _ALLOWED_FILTER_KEYS = {
        "domain",
        "doc_role",
        "content_type",
        "source",
        "source_path",
    }

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        qdrant_store: QdrantStore | None = None,
        answer_service: AnswerService | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        collection_name: str = "rag_chunks",
        vector_name: str = "dense",
        qdrant_url: str = "http://localhost:6333",
    ) -> None:
        self.embedding_service: EmbeddingService = (
            embedding_service or EmbeddingService()
        )

        probe_vector: list[float] = self.embedding_service.embed_text("dimension probe")

        self.qdrant_store: QdrantStore = qdrant_store or QdrantStore(
            url=qdrant_url,
            collection_name=collection_name,
            vector_name=vector_name,
            vector_size=len(probe_vector),
        )

        self.answer_service: AnswerService = answer_service or AnswerService()

        self.hybrid_retriever: HybridRetriever = hybrid_retriever or HybridRetriever(
            embedding_service=self.embedding_service,
            qdrant_store=self.qdrant_store,
        )

    def _retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any],
        retrieval_mode: RetrievalMode,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks using the selected retrieval mode."""
        if retrieval_mode == "dense":
            query_vector: list[float] = self.embedding_service.embed_text(query)

            results = self.qdrant_store.search(
                query_vector=query_vector,
                limit=top_k,
                filters=filters,
            )

            return [
                {
                    **result,
                    "retrieval_mode": "dense",
                }
                for result in results
            ]

        if retrieval_mode == "hybrid_rrf":
            return self.hybrid_retriever.search(
                query=query,
                limit=top_k,
                filters=filters,
            )

        raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")

    def answer(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        retrieval_mode: RetrievalMode = DEFAULT_RETRIEVAL_MODE,
    ) -> dict[str, Any]:
        """Retrieve relevant chunks and generate a grounded answer."""
        filters = filters or {}
        self._validate_filters(filters)

        run_id = self._make_run_id()

        logger.info("Running RAG query: %s", query)
        logger.info("top_k: %s", top_k)
        logger.info("filters: %s", filters)
        logger.info("retrieval_mode: %s", retrieval_mode)

        try:
            retrieved_chunks = self._retrieve(
                query=query,
                top_k=top_k,
                filters=filters,
                retrieval_mode=retrieval_mode,
            )
        except Exception as exc:
            raise RetrievalError("Failed to retrieve chunks.") from exc

        if not retrieved_chunks:
            raise EmptyRetrievalError("Retrieval returned no chunks.")

        try:
            answer: str = self.answer_service.answer(
                query=query,
                retrieved_chunks=retrieved_chunks,
            )
        except Exception as exc:
            raise GenerationError("Failed to generate grounded answer.") from exc

        return {
            "run_id": run_id,
            "query": query,
            "filters": filters,
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "answer": answer,
            "retrieved_chunks": self._format_retrieved_chunks(retrieved_chunks),
        }

    def readiness_check(self) -> dict[str, Any]:
        """Check whether required RAG dependencies are available."""
        try:
            qdrant_status: dict[str, Any] = self.qdrant_store.readiness_check()
        except Exception as exc:
            raise RetrievalError("Qdrant readiness check failed.") from exc

        return {
            "ready": qdrant_status["ready"],
            "dependencies": {
                "qdrant": qdrant_status,
            },
        }

    def _validate_filters(self, filters: dict[str, Any]) -> None:
        """Validate user-supplied metadata filters before retrieval."""
        invalid_keys: set[str] = set(filters) - self._ALLOWED_FILTER_KEYS

        if invalid_keys:
            allowed_keys: str = ", ".join(sorted(self._ALLOWED_FILTER_KEYS))
            invalid_keys_text: str = ", ".join(sorted(invalid_keys))

            raise InvalidFiltersError(
                f"Unsupported filter field(s): {invalid_keys_text}. "
                f"Allowed fields: {allowed_keys}."
            )

        invalid_value_keys: list[str] = [
            key for key, value in filters.items() if not isinstance(value, str)
        ]

        if invalid_value_keys:
            invalid_values_text: str = ", ".join(sorted(invalid_value_keys))

            raise InvalidFiltersError(
                f"Unsupported filter value type for field(s): {invalid_values_text}. "
                "Only string filter values are currently supported."
            )

    @staticmethod
    def _make_run_id() -> str:
        """Create a traceable run ID for one RAG request."""
        timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix: str = uuid.uuid4().hex[:8]
        return f"ask_{timestamp}_{suffix}"

    @staticmethod
    def _format_retrieved_chunks(
        retrieved_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert retrieved Qdrant results into a stable response shape."""
        formatted_chunks: list[dict[str, Any]] = []

        for index, result in enumerate(retrieved_chunks, start=1):
            payload: dict[str, Any] = result.get("payload") or {}

            formatted_chunks.append(
                {
                    "rank": index,
                    "score": result.get("score"),
                    "source": payload.get("source"),
                    "source_path": payload.get("source_path"),
                    "content_type": payload.get("content_type"),
                    "domain": payload.get("domain"),
                    "doc_role": payload.get("doc_role"),
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": payload.get("document_id"),
                    "section_id": payload.get("section_id"),
                    "text": payload.get("text"),
                }
            )

        return formatted_chunks
