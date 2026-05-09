import argparse
import logging
from pathlib import Path

from rag.stores.sqlite_lexical_store import SQLiteLexicalStore

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SQLite FTS lexical index from chunks."
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="Path to processed chunks JSONL file.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/indexes/lexical.sqlite"),
        help="Path to SQLite lexical index database.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    args = parse_args()

    logger.info("Building SQLite lexical index")
    logger.info("chunks_path: %s", args.chunks_path)
    logger.info("db_path: %s", args.db_path)

    store = SQLiteLexicalStore(db_path=args.db_path)
    store.rebuild_from_chunks(chunks_path=args.chunks_path)

    logger.info("SQLite lexical index build complete")


if __name__ == "__main__":
    main()
