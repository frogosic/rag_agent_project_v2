from pathlib import Path
import logging

from rag.config.content_types import load_content_types_config

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Check the content types configuration."""
    config_path = Path("config/content_types.yaml")
    config = load_content_types_config(config_path)

    logger.info("Loaded content types:")

    for name, content_type in config.content_types.items():
        logger.info("- %s", name)
        logger.info("  source_dir: %s", content_type.source_dir)
        logger.info("  file_patterns: %s", content_type.file_patterns)
        logger.info("  reader: %s", content_type.reader)
        logger.info("  sectioner: %s", content_type.sectioner)
        logger.info("  chunker: %s", content_type.chunker)


if __name__ == "__main__":
    main()
