import logging
from pathlib import Path

from rag.config.content_types import ContentTypeConfig
from rag.domain.documents import Document
from rag.ingestion.ids import make_document_id

logger = logging.getLogger(__name__)


class TextReader:
    """Reads a text-based source file into a Document.

    This reader handles plain text-like formats:
    - .txt
    - .md
    - .policy

    It does not section or chunk the document.
    """

    def read(
        self, source_path: Path, config_name: str, config: ContentTypeConfig
    ) -> Document:
        """Read the source file and create a Document."""
        logger.debug("Reading source file: %s", source_path)

        text = source_path.read_text(encoding="utf-8").strip()

        document_id = make_document_id(
            content_type=config_name,
            source_path=source_path,
        )

        metadata = dict(config.metadata)
        metadata.update(
            {
                "source": source_path.name,
                "source_path": source_path.as_posix(),
                "doc_format": source_path.suffix.lstrip("."),
            }
        )

        return Document(
            id=document_id,
            source_path=source_path.as_posix(),
            content_type=config_name,
            title=source_path.stem,
            text=text,
            metadata=metadata,
        )


def get_reader(reader_name: str) -> TextReader:
    """Factory function to get the appropriate reader based on the name."""
    if reader_name == "text":
        return TextReader()

    raise ValueError(f"Unknown reader: {reader_name}")
