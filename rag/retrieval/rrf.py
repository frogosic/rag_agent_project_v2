from typing import Any


def reciprocal_rank_fusion(
    ranked_result_lists: list[list[dict[str, Any]]],
    k: int = 60,
    limit: int = 5,
    weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Fuse ranked retrieval result lists using weighted Reciprocal Rank Fusion.

    RRF ignores incompatible raw scores and uses rank position instead.

    contribution = weight * (1 / (k + rank))
    """
    if weights is None:
        weights = [1.0] * len(ranked_result_lists)

    if len(weights) != len(ranked_result_lists):
        raise ValueError(
            "weights must have the same length as ranked_result_lists"
        )

    fused_by_chunk_id: dict[str, dict[str, Any]] = {}

    for result_list, weight in zip(ranked_result_lists, weights):
        for rank, result in enumerate(result_list, start=1):
            payload = result.get("payload") or {}
            chunk_id = payload.get("chunk_id")

            if not chunk_id:
                continue

            contribution = weight * (1 / (k + rank))

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
                    "weight": weight,
                    "original_score": result.get("score"),
                    "rrf_contribution": contribution,
                }
            )

    fused_results = list(fused_by_chunk_id.values())
    fused_results.sort(key=lambda result: result["score"], reverse=True)

    return fused_results[:limit]
