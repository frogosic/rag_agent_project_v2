import argparse
import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from rag.indexing.embedding_service import EmbeddingService
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.stores.qdrant_store import QdrantStore
from rag.stores.sqlite_lexical_store import SQLiteLexicalStore

logger = logging.getLogger(__name__)

DEFAULT_EVAL_PATH = Path("data/eval/retrieval_eval.yaml")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate dense retrieval against Qdrant."
    )

    parser.add_argument(
        "--eval-path",
        type=Path,
        default=DEFAULT_EVAL_PATH,
        help="Path to retrieval eval YAML file.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks to evaluate per query.",
    )

    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Path where retrieval eval results should be written.",
    )

    parser.add_argument(
        "--mode",
        choices=["dense", "lexical", "hybrid_rrf"],
        default="dense",
        help="Retrieval mode to evaluate.",
    )

    return parser.parse_args()


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    """Load evaluation cases from a YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Eval file must contain a top-level mapping.")

    cases = data.get("cases")

    if not isinstance(cases, list):
        raise ValueError("Eval file must contain a top-level 'cases' list.")

    return cases


def validate_case(case: dict[str, Any]) -> None:
    """Validate a single evaluation case."""
    required_fields = {"id", "query", "expected"}

    missing_fields = required_fields - set(case)

    if missing_fields:
        raise ValueError(
            f"Eval case is missing required fields: {sorted(missing_fields)}"
        )

    expected = case["expected"]

    if not isinstance(expected, dict):
        raise ValueError(f"Eval case {case['id']} has invalid expected block.")

    has_source_expectation = bool(expected.get("sources_any"))
    has_chunk_expectation = bool(expected.get("chunk_ids_any"))

    if not has_source_expectation and not has_chunk_expectation:
        raise ValueError(
            f"Eval case {case['id']} must define at least one of: "
            "expected.sources_any or expected.chunk_ids_any."
        )


def retrieve_for_eval(
    query: str,
    filters: dict[str, Any],
    top_k: int,
    mode: str,
    embedding_service: EmbeddingService,
    qdrant_store: QdrantStore,
    lexical_store: SQLiteLexicalStore,
    hybrid_retriever: HybridRetriever,
) -> list[dict[str, Any]]:
    """Retrieve chunks for one eval case using the selected retrieval mode."""
    if mode == "dense":
        query_vector = embedding_service.embed_text(query)
        results = qdrant_store.search(
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

    if mode == "lexical":
        return lexical_store.search(
            query=query,
            limit=top_k,
            filters=filters,
        )

    if mode == "hybrid_rrf":
        return hybrid_retriever.search(
            query=query,
            limit=top_k,
            filters=filters,
        )

    raise ValueError(f"Unsupported retrieval mode: {mode}")


def get_result_sources(results: list[dict[str, Any]]) -> list[str]:
    """Extract sources from retrieval results."""
    sources: list[str] = []

    for result in results:
        payload = result.get("payload") or {}
        source = payload.get("source")

        if source:
            sources.append(source)

    return sources


def get_result_chunk_ids(results: list[dict[str, Any]]) -> list[str]:
    """Extract chunk IDs from retrieval results."""
    chunk_ids: list[str] = []

    for result in results:
        payload = result.get("payload") or {}
        chunk_id = payload.get("chunk_id")

        if chunk_id:
            chunk_ids.append(chunk_id)

    return chunk_ids


def find_first_rank(
    actual_values: list[str],
    expected_values: set[str],
) -> int | None:
    """Return 1-based rank of the first expected value found in actual values."""
    for index, value in enumerate(actual_values, start=1):
        if value in expected_values:
            return index

    return None


def find_matching_values(
    actual_values: list[str],
    unwanted_values: set[str],
) -> list[str]:
    """Return unwanted values that appeared in actual values."""
    return [value for value in actual_values if value in unwanted_values]


def limit_values(values: list[str], limit: int | None) -> list[str]:
    """Return values limited to the first N items when a limit is provided."""
    if limit is None:
        return values

    return values[:limit]


def evaluate_case(
    case: dict[str, Any],
    results: list[dict[str, Any]],
    retrieval_mode: str = "dense_with_optional_filters",
) -> dict[str, Any]:
    """
    Evaluate a single case against retrieval results, returning a structured result.

    The evaluation logic checks for expected sources and chunk IDs in the results,
    as well as anti-signal sources that should not appear in the top K results.
    It also collects various metadata for analysis and debugging.
    """
    expected = case["expected"]
    anti_signals = case.get("anti_signals", {}) or {}
    anti_signal_check_top_k = anti_signals.get("check_top_k")
    warnings_config = case.get("warnings", {}) or {}
    filters = case.get("filters") or {}
    min_top_score = warnings_config.get("min_top_score")

    checks = case.get("checks", {}) or {}
    require_rank_lte = checks.get("require_rank_lte")

    expected_sources = set(expected.get("sources_any", []))
    expected_chunk_ids = set(expected.get("chunk_ids_any", []))
    anti_signal_sources = set(anti_signals.get("sources", []))

    result_sources = get_result_sources(results)
    result_chunk_ids = get_result_chunk_ids(results)

    source_rank = find_first_rank(result_sources, expected_sources)
    chunk_rank = find_first_rank(result_chunk_ids, expected_chunk_ids)

    source_hit = source_rank is not None
    chunk_hit = chunk_rank is not None

    anti_signal_result_sources = limit_values(
        result_sources,
        anti_signal_check_top_k,
    )

    matched_anti_signal_sources = find_matching_values(
        anti_signal_result_sources,
        anti_signal_sources,
    )

    anti_signal_hit = bool(matched_anti_signal_sources)

    top_score = results[0]["score"] if results else None

    warnings: list[str] = []

    if retrieval_mode == "dense":
        if (
            min_top_score is not None
            and top_score is not None
            and top_score < min_top_score
        ):
            warnings.append(
                f"top_score {top_score:.4f} is below min_top_score {min_top_score:.4f}"
            )

    if expected_chunk_ids:
        expected_hit = chunk_hit
        expected_rank = chunk_rank
    else:
        expected_hit = source_hit
        expected_rank = source_rank

    rank_requirement_met = True

    if require_rank_lte is not None:
        rank_requirement_met = (
            expected_rank is not None and expected_rank <= require_rank_lte
        )

    passed = expected_hit and rank_requirement_met and not anti_signal_hit

    return {
        "id": case["id"],
        "query": case["query"],
        "category": case.get("category"),
        "filters": filters,
        "passed": passed,
        "source_hit": source_hit,
        "chunk_hit": chunk_hit,
        "source_rank": source_rank,
        "chunk_rank": chunk_rank,
        "expected_sources": sorted(expected_sources),
        "expected_chunk_ids": sorted(expected_chunk_ids),
        "result_sources": result_sources,
        "result_chunk_ids": result_chunk_ids,
        "top_result_source": result_sources[0] if result_sources else None,
        "top_result_chunk_id": result_chunk_ids[0] if result_chunk_ids else None,
        "top_score": top_score,
        "warnings": warnings,
        "expected_hit": expected_hit,
        "anti_signal_hit": anti_signal_hit,
        "matched_anti_signal_sources": matched_anti_signal_sources,
        "anti_signal_check_top_k": anti_signal_check_top_k,
        "expected_rank": expected_rank,
        "require_rank_lte": require_rank_lte,
        "rank_requirement_met": rank_requirement_met,
    }


def average_rank(values: list[int | None]) -> float | None:
    """Calculate average rank, ignoring missing values."""
    ranks = [value for value in values if value is not None]

    if not ranks:
        return None

    return sum(ranks) / len(ranks)


def build_category_summary(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build summary grouped by eval category."""
    category_summary: dict[str, dict[str, Any]] = {}

    for result in results:
        category = result.get("category") or "uncategorized"

        if category not in category_summary:
            category_summary[category] = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "expected_misses": 0,
                "anti_signal_failures": 0,
                "rank_failures": 0,
            }

        bucket = category_summary[category]
        bucket["total"] += 1

        if result["passed"]:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1

        if not result["expected_hit"]:
            bucket["expected_misses"] += 1

        if result["anti_signal_hit"]:
            bucket["anti_signal_failures"] += 1

        if not result["rank_requirement_met"]:
            bucket["rank_failures"] += 1

    for bucket in category_summary.values():
        total = bucket["total"]
        bucket["pass_rate"] = bucket["passed"] / total * 100 if total else 0.0

    return category_summary


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build machine-readable evaluation summary."""
    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    failed = total - passed

    expected_misses = sum(1 for result in results if not result["expected_hit"])
    anti_signal_failures = sum(1 for result in results if result["anti_signal_hit"])
    rank_failures = sum(1 for result in results if not result["rank_requirement_met"])

    avg_source_rank = average_rank([result["source_rank"] for result in results])
    avg_chunk_rank = average_rank([result["chunk_rank"] for result in results])

    pass_rate = passed / total * 100 if total else 0.0

    category_summary = build_category_summary(results)
    warning_count = sum(len(result["warnings"]) for result in results)

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "expected_misses": expected_misses,
        "anti_signal_failures": anti_signal_failures,
        "rank_failures": rank_failures,
        "avg_source_rank": avg_source_rank,
        "avg_chunk_rank": avg_chunk_rank,
        "categories": category_summary,
        "warnings": warning_count,
    }


def log_case_result(result: dict[str, Any]) -> None:
    """Log the result of a single evaluation case."""
    status = "PASS" if result["passed"] else "FAIL"

    logger.info("[%s] %s", status, result["id"])
    logger.info("query: %s", result["query"])
    logger.info("category: %s", result["category"])
    logger.info("filters: %s", result["filters"])
    logger.info("top_score: %s", result["top_score"])
    logger.info("top_source: %s", result["top_result_source"])
    logger.info("source_hit: %s", result["source_hit"])
    logger.info("chunk_hit: %s", result["chunk_hit"])
    logger.info("source_rank: %s", result["source_rank"])
    logger.info("chunk_rank: %s", result["chunk_rank"])
    logger.info("expected_rank: %s", result["expected_rank"])
    logger.info("require_rank_lte: %s", result["require_rank_lte"])
    logger.info("rank_requirement_met: %s", result["rank_requirement_met"])
    logger.info("expected_hit: %s", result["expected_hit"])
    logger.info("anti_signal_hit: %s", result["anti_signal_hit"])
    logger.info("anti_signal_check_top_k: %s", result["anti_signal_check_top_k"])

    for warning in result["warnings"]:
        logger.warning("%s | %s", result["id"], warning)

    if not result["passed"]:
        logger.info("expected_sources: %s", result["expected_sources"])
        logger.info("actual_sources: %s", result["result_sources"])
        logger.info(
            "matched_anti_signal_sources: %s",
            result["matched_anti_signal_sources"],
        )


def log_summary(summary: dict[str, Any]) -> None:
    """Log the overall evaluation summary."""
    logger.info("=== Retrieval Eval Summary ===")
    logger.info("total: %s", summary["total"])
    logger.info("passed: %s", summary["passed"])
    logger.info("failed: %s", summary["failed"])
    logger.info("pass_rate: %.2f%%", summary["pass_rate"])
    logger.info("expected_misses: %s", summary["expected_misses"])
    logger.info("anti_signal_failures: %s", summary["anti_signal_failures"])
    logger.info("rank_failures: %s", summary["rank_failures"])
    logger.info("avg_source_rank: %s", summary["avg_source_rank"])
    logger.info("avg_chunk_rank: %s", summary["avg_chunk_rank"])
    logger.info("warnings: %s", summary["warnings"])

    logger.info("=== Category Summary ===")

    for category, category_result in summary["categories"].items():
        logger.info(
            "%s | total=%s passed=%s failed=%s pass_rate=%.2f%% "
            "expected_misses=%s anti_signal_failures=%s rank_failures=%s",
            category,
            category_result["total"],
            category_result["passed"],
            category_result["failed"],
            category_result["pass_rate"],
            category_result["expected_misses"],
            category_result["anti_signal_failures"],
            category_result["rank_failures"],
        )


def write_results(
    path: Path,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    retrieval_mode: str,
    top_k: int,
) -> None:
    """Write evaluation results to disk as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "summary": summary,
        "results": results,
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    logger.info("Wrote eval results to %s", path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    args = parse_args()

    cases = load_eval_cases(args.eval_path)

    for case in cases:
        validate_case(case)

    embedding_service = EmbeddingService()

    probe_vector = embedding_service.embed_text("dimension probe")

    qdrant_store = QdrantStore(
        url="http://localhost:6333",
        collection_name="rag_chunks",
        vector_name="dense",
        vector_size=len(probe_vector),
    )

    lexical_store = SQLiteLexicalStore(db_path=Path("data/indexes/lexical.sqlite"))

    hybrid_retriever = HybridRetriever(
        embedding_service=embedding_service,
        qdrant_store=qdrant_store,
        lexical_store=lexical_store,
    )

    eval_results: list[dict[str, Any]] = []

    for case in cases:
        logger.info("Evaluating case: %s", case["id"])

        filters = case.get("filters") or {}

        search_results = retrieve_for_eval(
            query=case["query"],
            filters=filters,
            top_k=args.top_k,
            mode=args.mode,
            embedding_service=embedding_service,
            qdrant_store=qdrant_store,
            lexical_store=lexical_store,
            hybrid_retriever=hybrid_retriever,
        )

        eval_result = evaluate_case(
            case=case,
            results=search_results,
            retrieval_mode=args.mode,
        )
        
        eval_results.append(eval_result)
        log_case_result(eval_result)

    summary = build_summary(eval_results)
    log_summary(summary)

    results_path = args.results_path or Path(
        f"data/eval/results/retrieval_eval_{args.mode}.json"
    )

    write_results(
        path=results_path,
        summary=summary,
        results=eval_results,
        retrieval_mode=args.mode,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
