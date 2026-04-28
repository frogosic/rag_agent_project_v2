import logging
from pathlib import Path

from rag.config.content_types import ContentTypesConfig
from rag.domain.documents import Chunk, Document, Section
from rag.ingestion.chunkers import get_chunker
from rag.ingestion.readers import get_reader
from rag.ingestion.sectioners import get_sectioner

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Runs configured ingestion from raw files to chunks."""

    def __init__(self, config: ContentTypesConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root

    def run(self) -> list[Chunk]:
        """Run the ingestion pipeline and return all chunks."""
        all_chunks: list[Chunk] = []

        for content_type_name, content_type_config in self.config.content_types.items():
            logger.info("Ingesting content type: %s", content_type_name)

            source_paths = self._discover_source_paths(
                content_type_config.source_dir, content_type_config.file_patterns
            )

            logger.info(
                "Discovered %s source files for content type: %s",
                len(source_paths),
                content_type_name,
            )

            reader = get_reader(content_type_config.reader)
            sectioner = get_sectioner(content_type_config.sectioner)
            chunker = get_chunker(content_type_config.chunker)

            for source_path in source_paths:
                logger.debug("Processing file: %s", source_path)

                document: Document = reader.read(
                    source_path=source_path,
                    config_name=content_type_name,
                    config=content_type_config,
                )

                sections: list[Section] = sectioner.section(
                    document=document,
                    config=content_type_config,
                )

                chunks: list[Chunk] = chunker.chunk(
                    sections=sections,
                    config=content_type_config,
                )

                logger.debug(
                    "Processed file=%s sections=%s chunks=%s",
                    source_path,
                    len(sections),
                    len(chunks),
                )

                all_chunks.extend(chunks)

        logger.info("Ingestion completed. Total chunks: %s", len(all_chunks))
        return all_chunks

    def _discover_source_paths(
        self,
        source_dir: Path,
        file_patterns: list[str],
    ) -> list[Path]:
        """Discover source files based on the configured directory and file patterns."""
        absolute_source_dir = self.project_root / source_dir

        if not absolute_source_dir.exists():
            logger.warning("Source directory does not exist: %s", absolute_source_dir)
            return []

        discovered: set[Path] = set()

        for pattern in file_patterns:
            matched_paths = absolute_source_dir.glob(pattern)

            for path in matched_paths:
                if path.is_file():
                    discovered.add(path)

        return sorted(discovered)
