from typing import Any

from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.query_classifier import is_exact_lookup_query
from rag.retrieval.reranker import Reranker


RetrievalResult = dict[str, Any]


class RerankedHybridRetriever:
    def __init__(
        self,
        *,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        candidate_multiplier: int = 3,
        min_candidate_limit: int = 10,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.candidate_multiplier = candidate_multiplier
        self.min_candidate_limit = min_candidate_limit

    def search(
        self,
        *,
        query: str,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        candidate_limit = max(
            limit * self.candidate_multiplier,
            self.min_candidate_limit,
        )

        candidates = self.hybrid_retriever.search(
            query=query,
            limit=candidate_limit,
            filters=filters or {},
        )

        if is_exact_lookup_query(query):
            return candidates[:limit], {
                "rerank_applied": False,
                "rerank_bypass_reason": "exact_lookup_query",
                "candidate_limit": candidate_limit,
            }

        reranked_results = self.reranker.rerank(
            query=query,
            candidates=candidates,
            limit=limit,
        )

        return reranked_results, {
            "rerank_applied": True,
            "rerank_bypass_reason": None,
            "candidate_limit": candidate_limit,
        }
