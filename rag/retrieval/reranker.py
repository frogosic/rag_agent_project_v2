from typing import Any, Protocol


RetrievalResult = dict[str, Any]


class Reranker(Protocol):
    def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievalResult],
        limit: int,
    ) -> list[RetrievalResult]:
        """
        Reorder retrieved candidates for the given query.

        Implementations should return at most `limit` candidates.
        """
        ...


class NoOpReranker:
    def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievalResult],
        limit: int,
    ) -> list[RetrievalResult]:
        if limit < 1:
            return []

        return candidates[:limit]


class ScoreReranker:
    def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievalResult],
        limit: int,
    ) -> list[RetrievalResult]:
        """
        Baseline score reranker.

        Sorts by existing retrieval score descending.
        This is not semantically smarter than retrieval; it only proves
        that the reranker can reorder candidates.
        """
        if limit < 1:
            return []

        return sorted(
            candidates,
            key=lambda candidate: candidate.get("score", 0.0),
            reverse=True,
        )[:limit]


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install it before using this reranker."
            ) from exc

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievalResult],
        limit: int,
    ) -> list[RetrievalResult]:
        if limit < 1:
            return []

        if not candidates:
            return []

        pairs = [(query, self._candidate_text(candidate)) for candidate in candidates]

        scores = self.model.predict(pairs)

        scored_candidates = []
        for candidate, rerank_score in zip(candidates, scores):
            enriched = dict(candidate)
            enriched["rerank_score"] = float(rerank_score)
            scored_candidates.append(enriched)

        return sorted(
            scored_candidates,
            key=lambda candidate: candidate["rerank_score"],
            reverse=True,
        )[:limit]

    @staticmethod
    def _candidate_text(candidate: RetrievalResult) -> str:
        text = candidate.get("text")
        if isinstance(text, str):
            return text

        payload = candidate.get("payload")
        if isinstance(payload, dict):
            payload_text = payload.get("text")
            if isinstance(payload_text, str):
                return payload_text

        return ""
