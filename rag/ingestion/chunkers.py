import logging
import re

from rag.config.content_types import ContentTypeConfig
from rag.domain.documents import Chunk, Section
from rag.ingestion.ids import make_chunk_id

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Very rough token estimate.

    Good enough for ingestion baseline.
    Later we can replace this with tokenizer-specific counting.
    """
    return max(1, len(text.split()))


class SingleChunker:
    """Creates one chunk per section."""

    def chunk(self, sections: list[Section], config: ContentTypeConfig) -> list[Chunk]:
        """Treat each section as a single chunk."""
        chunks: list[Chunk] = []

        for section in sections:
            text: str = section.text.strip()

            if not text:
                logger.debug("Skipping empty section: %s", section.id)
                continue

            chunks.append(
                Chunk(
                    id=make_chunk_id(section_id=section.id, ordinal=0, text=text),
                    document_id=section.document_id,
                    section_id=section.id,
                    ordinal=0,
                    text=text,
                    metadata={
                        **section.metadata,
                        "chunk_ordinal": 0,
                        "chunk_strategy": "single",
                    },
                )
            )

        return chunks


class ParagraphChunker:
    """Splits each section into paragraph-based chunks."""

    _PARAGRAPH_SPLIT_RE: re.Pattern[str] = re.compile(r"\n\s*\n+")

    def chunk(self, sections: list[Section], config: ContentTypeConfig) -> list[Chunk]:
        """Split sections into chunks based on paragraphs, respecting max_tokens and overlap."""
        chunks: list[Chunk] = []
        max_tokens = config.chunking.max_tokens

        for section in sections:
            paragraphs = [
                paragraph.strip()
                for paragraph in self._PARAGRAPH_SPLIT_RE.split(section.text)
                if paragraph.strip()
            ]

            current_parts: list[str] = []
            current_tokens = 0
            chunk_ordinal = 0

            for paragraph in paragraphs:
                paragraph_tokens = estimate_tokens(paragraph)

                if current_parts and current_tokens + paragraph_tokens > max_tokens:
                    chunk_text = "\n\n".join(current_parts).strip()
                    chunks.append(
                        self._build_chunk(
                            section=section,
                            ordinal=chunk_ordinal,
                            text=chunk_text,
                            strategy="paragraph",
                        )
                    )
                    chunk_ordinal += 1
                    current_parts = []
                    current_tokens = 0

                current_parts.append(paragraph)
                current_tokens += paragraph_tokens

            if current_parts:
                chunk_text = "\n\n".join(current_parts).strip()
                chunks.append(
                    self._build_chunk(
                        section=section,
                        ordinal=chunk_ordinal,
                        text=chunk_text,
                        strategy="paragraph",
                    )
                )

        return chunks

    def _build_chunk(
        self,
        section: Section,
        ordinal: int,
        text: str,
        strategy: str,
    ) -> Chunk:
        """Helper to build a Chunk with a stable ID."""
        return Chunk(
            id=make_chunk_id(section_id=section.id, ordinal=ordinal, text=text),
            document_id=section.document_id,
            section_id=section.id,
            ordinal=ordinal,
            text=text,
            metadata={
                **section.metadata,
                "chunk_ordinal": ordinal,
                "chunk_strategy": strategy,
            },
        )


class SectionWindowChunker:
    """Chunks sections while preserving section boundaries.

    If a section is below max_tokens, it becomes one chunk.
    If it is above max_tokens, it falls back to paragraph chunking.
    """

    def __init__(self) -> None:
        self._paragraph_chunker = ParagraphChunker()

    def chunk(self, sections: list[Section], config: ContentTypeConfig) -> list[Chunk]:
        chunks: list[Chunk] = []

        for section in sections:
            token_count: int = estimate_tokens(section.text)

            if token_count <= config.chunking.max_tokens:
                chunks.append(
                    Chunk(
                        id=make_chunk_id(
                            section_id=section.id,
                            ordinal=0,
                            text=section.text,
                        ),
                        document_id=section.document_id,
                        section_id=section.id,
                        ordinal=0,
                        text=section.text,
                        metadata={
                            **section.metadata,
                            "chunk_ordinal": 0,
                            "chunk_strategy": "section_window",
                            "estimated_tokens": token_count,
                        },
                    )
                )
                continue

            logger.debug(
                "Section exceeds max_tokens; falling back to paragraph chunking: "
                "section_id=%s estimated_tokens=%s max_tokens=%s",
                section.id,
                token_count,
                config.chunking.max_tokens,
            )

            section_chunks = self._paragraph_chunker.chunk([section], config)

            for chunk in section_chunks:
                chunk.metadata["chunk_strategy"] = "section_window_paragraph_fallback"
                chunk.metadata["estimated_tokens"] = estimate_tokens(chunk.text)

            chunks.extend(section_chunks)

        return chunks


def get_chunker(
    chunker_name: str,
) -> SingleChunker | ParagraphChunker | SectionWindowChunker:
    """Factory function to get the appropriate chunker based on the name."""
    if chunker_name == "single":
        return SingleChunker()

    if chunker_name == "paragraph":
        return ParagraphChunker()

    if chunker_name == "section_window":
        return SectionWindowChunker()

    raise ValueError(f"Unknown chunker: {chunker_name}")
