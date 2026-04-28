import json
import logging
from dataclasses import asdict
from pathlib import Path

from rag.config.content_types import load_content_types_config
from rag.ingestion.pipeline import IngestionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Run ingestion and write chunks to an inspectable JSONL file."""
    project_root = Path.cwd()

    config_path = project_root / "config/content_types.yaml"
    output_path = project_root / "data/processed/chunks.jsonl"

    config = load_content_types_config(config_path)

    pipeline = IngestionPipeline(
        config=config,
        project_root=project_root,
    )

    chunks = pipeline.run()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    logger.info("Wrote %s chunks to %s", len(chunks), output_path)


if __name__ == "__main__":
    main()
