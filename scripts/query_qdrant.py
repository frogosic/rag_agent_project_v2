import argparse
import logging
from typing import Any

from rag.indexing.embedding_service import EmbeddingService
from rag.indexing.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a dense vector search against the local Qdrant collection."
    )

    parser.add_argument(
        "query",
        type=str,
        help="Natural-language query to search for.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return.",
    )

    return parser.parse_args()


def format_result(index: int, result: dict[str, Any]) -> str:
    payload = result["payload"] or {}

    text = payload.get("text", "")
    source = payload.get("source")
    source_path = payload.get("source_path")
    content_type = payload.get("content_type")
    domain = payload.get("domain")
    doc_role = payload.get("doc_role")
    chunk_id = payload.get("chunk_id")

    preview = text.replace("\n", " ").strip()

    if len(preview) > 500:
        preview = f"{preview[:500]}..."

    return (
        f"\n--- Result {index} ---\n"
        f"score: {result['score']:.4f}\n"
        f"chunk_id: {chunk_id}\n"
        f"source: {source}\n"
        f"source_path: {source_path}\n"
        f"content_type: {content_type}\n"
        f"domain: {domain}\n"
        f"doc_role: {doc_role}\n"
        f"text:\n{preview}\n"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    args = parse_args()

    embedding_service = EmbeddingService()
    query_vector = embedding_service.embed_text(args.query)

    store = QdrantStore(
        url="http://localhost:6333",
        collection_name="rag_chunks",
        vector_name="dense",
        vector_size=len(query_vector),
    )

    results = store.search(
        query_vector=query_vector,
        limit=args.top_k,
    )

    logger.info("Query: %s", args.query)
    logger.info("Returned %s results", len(results))

    for index, result in enumerate(results, start=1):
        print(format_result(index, result))


if __name__ == "__main__":
    main()
