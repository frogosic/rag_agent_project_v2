import argparse
import logging
from pathlib import Path
from typing import Any

from rag.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid retrieval: dense Qdrant + SQLite lexical + RRF."
    )
    parser.add_argument("query", help="Query text.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help="Number of candidates to retrieve from each retriever before RRF.",
    )
    parser.add_argument("--domain", default=None)
    parser.add_argument("--doc-role", default=None)
    parser.add_argument("--content-type", default=None)
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
    )
    parser.add_argument(
        "--collection-name",
        default="rag_chunks",
    )
    parser.add_argument(
        "--vector-name",
        default="dense",
    )
    parser.add_argument(
        "--lexical-db-path",
        type=Path,
        default=Path("data/indexes/lexical.sqlite"),
    )
    return parser.parse_args()


def build_filters(args: argparse.Namespace) -> dict[str, Any]:
    filters: dict[str, Any] = {}

    if args.domain:
        filters["domain"] = args.domain

    if args.doc_role:
        filters["doc_role"] = args.doc_role

    if args.content_type:
        filters["content_type"] = args.content_type

    return filters


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    args = parse_args()
    filters = build_filters(args)

    retriever = HybridRetriever(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection_name,
        vector_name=args.vector_name,
        lexical_db_path=args.lexical_db_path,
    )

    results = retriever.search(
        query=args.query,
        limit=args.top_k,
        filters=filters,
        candidate_limit=args.candidate_limit,
    )

    logger.info("Query: %s", args.query)
    logger.info("Filters: %s", filters)
    logger.info("Results: %s", len(results))

    for index, result in enumerate(results, start=1):
        payload = result["payload"]

        print("=" * 100)
        print(f"Rank: {index}")
        print(f"RRF score: {result['score']:.6f}")
        print(f"Retrieval mode: {result['retrieval_mode']}")
        print(f"Source modes: {result.get('source_modes')}")
        print(f"Source: {payload.get('source')}")
        print(f"Source path: {payload.get('source_path')}")
        print(f"Content type: {payload.get('content_type')}")
        print(f"Domain: {payload.get('domain')}")
        print(f"Doc role: {payload.get('doc_role')}")
        print(f"Chunk ID: {payload.get('chunk_id')}")
        print("-" * 100)

        print("RRF details:")
        for detail in result.get("rrf_details", []):
            print(
                f"  - mode={detail.get('retrieval_mode')} "
                f"rank={detail.get('rank')} "
                f"original_score={detail.get('original_score')} "
                f"rrf={detail.get('rrf_contribution'):.6f}"
            )

        print("-" * 100)
        print(payload.get("text"))


if __name__ == "__main__":
    main()
