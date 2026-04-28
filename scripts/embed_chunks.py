import json
import logging
from pathlib import Path
from typing import Any

from rag.indexing.embedding_service import EmbeddingService

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

logger = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write dictionaries to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """Embed processed chunks and write embedded chunks to JSONL."""
    project_root = Path.cwd()

    input_path = project_root / "data/processed/chunks.jsonl"
    output_path = project_root / "data/processed/embedded_chunks.jsonl"

    logger.info("Reading chunks from %s", input_path)
    chunks = read_jsonl(input_path)

    texts = [chunk["text"] for chunk in chunks]

    embedding_service = EmbeddingService()
    embeddings = embedding_service.embed_texts(texts)

    if len(chunks) != len(embeddings):
        raise RuntimeError(
            f"Chunk/embedding count mismatch: chunks={len(chunks)} embeddings={len(embeddings)}"
        )

    embedded_chunks: list[dict[str, Any]] = []

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        embedded_chunks.append(
            {
                **chunk,
                "embedding": embedding,
                "embedding_model": embedding_service.model_name,
                "embedding_dimension": len(embedding),
            }
        )

    write_jsonl(embedded_chunks, output_path)

    logger.info("Wrote %s embedded chunks to %s", len(embedded_chunks), output_path)


if __name__ == "__main__":
    main()
