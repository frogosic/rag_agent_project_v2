from typing import Any


def reciprocal_rank_fusion(
    ranked_result_lists: list[list[dict[str, Any]]],
    k: int = 60,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked retrieval result lists using Reciprocal Rank Fusion.

    RRF ignores raw scores from different retrievers and uses rank position instead.

    score = sum(1 / (k + rank))
    """
    fused_by_chunk_id: dict[str, dict[str, Any]] = {}

    for result_list in ranked_result_lists:
        for rank, result in enumerate(result_list, start=1):
            payload = result.get("payload") or {}
            chunk_id = payload.get("chunk_id")

            if not chunk_id:
                continue

            contribution = 1 / (k + rank)

            if chunk_id not in fused_by_chunk_id:
                fused_by_chunk_id[chunk_id] = {
                    "score": 0.0,
                    "payload": payload,
                    "retrieval_mode": "hybrid_rrf",
                    "source_modes": [],
                    "rrf_details": [],
                }

            fused_result = fused_by_chunk_id[chunk_id]
            fused_result["score"] += contribution
            fused_result["source_modes"].append(result.get("retrieval_mode"))
            fused_result["rrf_details"].append(
                {
                    "retrieval_mode": result.get("retrieval_mode"),
                    "rank": rank,
                    "original_score": result.get("score"),
                    "rrf_contribution": contribution,
                }
            )

    fused_results: list[dict[str, Any]] = list(fused_by_chunk_id.values())
    fused_results.sort(key=lambda result: result["score"], reverse=True)

    return fused_results[:limit]
