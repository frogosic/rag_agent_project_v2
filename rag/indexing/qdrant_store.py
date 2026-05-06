import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)


DEFAULT_COLLECTION_NAME = "rag_chunks"
DEFAULT_VECTOR_NAME = "dense"
DEFAULT_VECTOR_SIZE = 384


class QdrantStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_name: str = DEFAULT_VECTOR_NAME,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> None:
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.vector_size = vector_size

    def recreate_collection(self) -> None:
        """
        Recreate the collection from scratch.

        This is intentionally destructive and deterministic for the baseline:
        same embedded_chunks.jsonl input -> same Qdrant collection state.
        """
        if self.client.collection_exists(self.collection_name):
            logger.info("Deleting existing Qdrant collection: %s", self.collection_name)
            self.client.delete_collection(collection_name=self.collection_name)

        logger.info(
            "Creating Qdrant collection: %s | vector=%s | size=%s | distance=cosine",
            self.collection_name,
            self.vector_name,
            self.vector_size,
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.vector_name: VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                )
            },
        )

    def upsert_embedded_chunks(
        self,
        embedded_chunks: list[dict[str, Any]],
        batch_size: int = 64,
    ) -> int:
        total = 0

        for start in range(0, len(embedded_chunks), batch_size):
            batch = embedded_chunks[start : start + batch_size]
            points = [self._chunk_to_point(chunk) for chunk in batch]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            total += len(points)
            logger.info("Upserted %s/%s chunks", total, len(embedded_chunks))

        return total

    def _chunk_to_point(self, chunk: dict[str, Any]) -> PointStruct:
        chunk_id = chunk["id"]
        embedding = chunk["embedding"]

        if len(embedding) != self.vector_size:
            raise ValueError(
                f"Embedding dimension mismatch for chunk_id={chunk_id}. "
                f"Expected {self.vector_size}, got {len(embedding)}."
            )

        point_id = self._make_point_id(chunk_id)

        payload = self._build_payload(chunk)

        return PointStruct(
            id=point_id,
            vector={
                self.vector_name: embedding,
            },
            payload=payload,
        )

    @staticmethod
    def _make_point_id(chunk_id: str) -> str:
        """
        Create a deterministic UUID from the stable chunk ID.

        This avoids random Qdrant IDs while still satisfying Qdrant's point ID shape.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    @staticmethod
    def _build_payload(chunk: dict[str, Any]) -> dict[str, Any]:
        metadata = chunk.get("metadata", {})

        return {
            "text": chunk["text"],
            "chunk_id": chunk["id"],
            "document_id": chunk["document_id"],
            "section_id": chunk["section_id"],
            "ordinal": chunk["ordinal"],
            "embedding_model": chunk.get("embedding_model"),
            "embedding_dimension": chunk.get("embedding_dimension"),
            "source": metadata.get("source"),
            "source_path": metadata.get("source_path"),
            "content_type": metadata.get("content_type"),
            "domain": metadata.get("domain"),
            "doc_role": metadata.get("doc_role"),
            "metadata": metadata,
        }

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> Filter | None:
        if not filters:
            return None

        conditions = [
            FieldCondition(
                key=key,
                match=MatchValue(value=value),
            )
            for key, value in filters.items()
        ]

        return Filter(must=conditions)  # type: ignore

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"Query vector dimension mismatch. "
                f"Expected {self.vector_size}, got {len(query_vector)}."
            )

        query_filter = self._build_filter(filters)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=self.vector_name,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload,
            }
            for point in results.points
        ]
