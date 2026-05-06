import logging
import uuid
from typing import Any
from datetime import datetime, timezone

from rag.generation.answer_service import AnswerService
from rag.indexing.embedding_service import EmbeddingService
from rag.indexing.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class RAGService:
    """Application-level service for retrieval-augmented answering."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        qdrant_store: QdrantStore | None = None,
        answer_service: AnswerService | None = None,
        collection_name: str = "rag_chunks",
        vector_name: str = "dense",
        qdrant_url: str = "http://localhost:6333",
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()

        probe_vector = self.embedding_service.embed_text("dimension probe")

        self.qdrant_store = qdrant_store or QdrantStore(
            url=qdrant_url,
            collection_name=collection_name,
            vector_name=vector_name,
            vector_size=len(probe_vector),
        )

        self.answer_service = answer_service or AnswerService()

    @staticmethod
    def _make_run_id() -> str:
        """Create a traceable run ID for one RAG request."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = uuid.uuid4().hex[:8]
        return f"ask_{timestamp}_{suffix}"

    def answer(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant chunks and generate a grounded answer."""
        filters = filters or {}
        run_id = self._make_run_id()

        logger.info("Running RAG query: %s", query)
        logger.info("top_k: %s", top_k)
        logger.info("filters: %s", filters)

        query_vector = self.embedding_service.embed_text(query)

        retrieved_chunks = self.qdrant_store.search(
            query_vector=query_vector,
            limit=top_k,
            filters=filters,
        )

        answer = self.answer_service.answer(
            query=query,
            retrieved_chunks=retrieved_chunks,
        )

        return {
            "run_id": run_id,
            "query": query,
            "filters": filters,
            "top_k": top_k,
            "retrieval_mode": "dense_with_optional_filters",
            "answer": answer,
            "retrieved_chunks": self._format_retrieved_chunks(retrieved_chunks),
        }

    @staticmethod
    def _format_retrieved_chunks(
        retrieved_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert retrieved Qdrant results into a stable response shape."""
        formatted_chunks: list[dict[str, Any]] = []

        for index, result in enumerate(retrieved_chunks, start=1):
            payload = result.get("payload") or {}

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
