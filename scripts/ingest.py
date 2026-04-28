import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from rag.config.content_types import load_content_types_config
from rag.domain.documents import Chunk
from rag.ingestion.pipeline import IngestionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

logger = logging.getLogger(__name__)


def write_chunks_jsonl(chunks: list[Chunk], output_path: Path) -> None:
    """Write chunks to JSONL for inspection and later indexing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def log_ingestion_summary(chunks: list[Chunk]) -> None:
    """Log a compact summary of generated chunks."""
    content_type_counts = Counter(
        chunk.metadata.get("content_type", "unknown") for chunk in chunks
    )

    doc_role_counts = Counter(
        chunk.metadata.get("doc_role", "unknown") for chunk in chunks
    )

    logger.info("Chunk count by content_type:")
    for content_type, count in sorted(content_type_counts.items()):
        logger.info("  %s: %s", content_type, count)

    logger.info("Chunk count by doc_role:")
    for doc_role, count in sorted(doc_role_counts.items()):
        logger.info("  %s: %s", doc_role, count)


def main() -> None:
    """Run ingestion and write processed chunks."""
    project_root = Path.cwd()

    config_path = project_root / "config/content_types.yaml"
    output_path = project_root / "data/processed/chunks.jsonl"

    logger.info("Loading content types config: %s", config_path)

    config = load_content_types_config(config_path)

    pipeline = IngestionPipeline(
        config=config,
        project_root=project_root,
    )

    chunks = pipeline.run()

    write_chunks_jsonl(
        chunks=chunks,
        output_path=output_path,
    )

    log_ingestion_summary(chunks)

    logger.info("Wrote %s chunks to %s", len(chunks), output_path)


if __name__ == "__main__":
    main()
