import argparse
import logging
from pathlib import Path
from typing import Any

from rag.stores.sqlite_lexical_store import SQLiteLexicalStore

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the SQLite FTS lexical index.")
    parser.add_argument("query", help="Query text.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--doc-role", default=None)
    parser.add_argument("--content-type", default=None)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/indexes/lexical.sqlite"),
        help="Path to SQLite lexical index database.",
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

    store = SQLiteLexicalStore(db_path=args.db_path)

    results = store.search(
        query=args.query,
        limit=args.top_k,
        filters=filters,
    )

    logger.info("Query: %s", args.query)
    logger.info("Filters: %s", filters)
    logger.info("Results: %s", len(results))

    for index, result in enumerate(results, start=1):
        payload = result["payload"]

        print("=" * 100)
        print(f"Rank: {index}")
        print(f"Score: {result['score']:.6f}")
        print(f"Retrieval mode: {result['retrieval_mode']}")
        print(f"Source: {payload.get('source')}")
        print(f"Source path: {payload.get('source_path')}")
        print(f"Content type: {payload.get('content_type')}")
        print(f"Domain: {payload.get('domain')}")
        print(f"Doc role: {payload.get('doc_role')}")
        print(f"Chunk ID: {payload.get('chunk_id')}")
        print("-" * 100)
        print(payload.get("text"))


if __name__ == "__main__":
    main()
