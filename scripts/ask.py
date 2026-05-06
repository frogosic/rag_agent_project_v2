import argparse
import logging
from pathlib import Path
from typing import Any

from rag.application.rag_service import RAGService
from rag.application.run_store import RunStore

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = Path("data/runs/ask_runs")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ask a question against the local RAG index."
    )

    parser.add_argument(
        "query",
        type=str,
        help="Question to answer.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks to use as context.",
    )

    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Optional metadata filter for domain.",
    )

    parser.add_argument(
        "--doc-role",
        type=str,
        default=None,
        help="Optional metadata filter for doc_role.",
    )

    parser.add_argument(
        "--content-type",
        type=str,
        default=None,
        help="Optional metadata filter for content_type.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory where ask run JSON files should be written.",
    )

    return parser.parse_args()


def build_filters(args: argparse.Namespace) -> dict[str, Any]:
    """Build metadata filters from CLI arguments."""
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

    rag_service = RAGService()

    result = rag_service.answer(
        query=args.query,
        top_k=args.top_k,
        filters=filters,
    )

    run_store = RunStore(runs_dir=args.output_dir)
    output_path = run_store.write_ask_run(result=result)

    logger.info("Generated answer successfully.")
    logger.info("Wrote ask run to %s", output_path)


if __name__ == "__main__":
    main()
