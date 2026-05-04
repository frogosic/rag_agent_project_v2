import json
import logging
from pathlib import Path
from typing import Any

from rag.indexing.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

EMBEDDED_CHUNKS_PATH = Path("data/processed/embedded_chunks.jsonl")


def load_embedded_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Embedded chunks file not found: {path}. "
            "Run scripts/embed_chunks.py first."
        )

    chunks: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}"
                ) from exc

    return chunks


def validate_embedded_chunks(chunks: list[dict[str, Any]]) -> None:
    required_fields = {
        "id",
        "document_id",
        "section_id",
        "ordinal",
        "text",
        "metadata",
        "embedding",
        "embedding_model",
        "embedding_dimension",
    }

    for index, chunk in enumerate(chunks):
        missing_fields = required_fields - set(chunk)

        if missing_fields:
            raise ValueError(
                f"Chunk at index {index} is missing required fields: "
                f"{sorted(missing_fields)}"
            )

        if not isinstance(chunk["embedding"], list):
            raise ValueError(f"Chunk at index {index} has invalid embedding field.")

        if chunk["embedding_dimension"] != len(chunk["embedding"]):
            raise ValueError(
                f"Chunk at index {index} has embedding_dimension="
                f"{chunk['embedding_dimension']} but actual vector length="
                f"{len(chunk['embedding'])}."
            )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    logger.info("Loading embedded chunks from %s", EMBEDDED_CHUNKS_PATH)
    chunks = load_embedded_chunks(EMBEDDED_CHUNKS_PATH)

    logger.info("Loaded %s embedded chunks", len(chunks))
    validate_embedded_chunks(chunks)

    if not chunks:
        raise ValueError("No embedded chunks found. Nothing to index.")

    embedding_dimension = chunks[0]["embedding_dimension"]

    store = QdrantStore(
        url="http://localhost:6333",
        collection_name="rag_chunks",
        vector_name="dense",
        vector_size=embedding_dimension,
    )

    store.recreate_collection()
    indexed_count = store.upsert_embedded_chunks(chunks)

    logger.info("Indexed %s chunks into Qdrant collection 'rag_chunks'", indexed_count)


if __name__ == "__main__":
    main()
