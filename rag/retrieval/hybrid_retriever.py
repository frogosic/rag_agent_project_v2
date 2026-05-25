import logging
from pathlib import Path
from typing import Any

from rag.indexing.embedding_service import EmbeddingService
from rag.stores.qdrant_store import QdrantStore
from rag.retrieval.rrf import reciprocal_rank_fusion
from rag.stores.sqlite_lexical_store import SQLiteLexicalStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever using dense Qdrant search + SQLite lexical search + RRF."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        qdrant_store: QdrantStore | None = None,
        lexical_store: SQLiteLexicalStore | None = None,
        collection_name: str = "rag_chunks",
        vector_name: str = "dense",
        qdrant_url: str = "http://localhost:6333",
        lexical_db_path: Path = Path("data/indexes/lexical.sqlite"),
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        lexical_weight: float = 0.6,
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

        self.lexical_store: SQLiteLexicalStore = lexical_store or SQLiteLexicalStore(
            db_path=lexical_db_path,
        )

        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        candidate_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run dense and lexical retrieval, then fuse results with RRF."""
        filters = filters or {}
        candidate_limit = candidate_limit or max(limit * 3, 10)

        logger.info(
            "Running hybrid retrieval | query=%s | limit=%s | candidate_limit=%s | filters=%s",
            query,
            limit,
            candidate_limit,
            filters,
        )

        dense_results: list[dict[str, Any]] = self._dense_search(
            query=query,
            limit=candidate_limit,
            filters=filters,
        )
        lexical_results: list[dict[str, Any]] = self._lexical_search(
            query=query,
            limit=candidate_limit,
            filters=filters,
        )

        fused_results = reciprocal_rank_fusion(
            ranked_result_lists=[
                dense_results,
                lexical_results,
            ],
            k=self.rrf_k,
            limit=limit,
            weights=[
                self.dense_weight,
                self.lexical_weight,
            ],
        )

        logger.info(
            "Hybrid retrieval complete | dense=%s | lexical=%s | fused=%s",
            len(dense_results),
            len(lexical_results),
            len(fused_results),
        )

        return fused_results

    def _dense_search(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run dense vector retrieval."""
        query_vector: list[float] = self.embedding_service.embed_text(query)

        results: list[dict[str, Any]] = self.qdrant_store.search(
            query_vector=query_vector,
            limit=limit,
            filters=filters,
        )

        return [
            {
                **result,
                "retrieval_mode": "dense",
            }
            for result in results
        ]

    def _lexical_search(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run SQLite FTS lexical retrieval."""
        return self.lexical_store.search(
            query=query,
            limit=limit,
            filters=filters,
        )
