import logging
import re

from markdown_it import MarkdownIt
from markdown_it.token import Token

from rag.config.content_types import ContentTypeConfig
from rag.domain.documents import Document, Section
from rag.ingestion.ids import make_section_id

logger = logging.getLogger(__name__)


class PlaintextSectioner:
    """Creates one section from the whole document."""

    def section(self, document: Document, _config: ContentTypeConfig) -> list[Section]:
        """Treat the entire document as a single section."""
        text = document.text.strip()

        if not text:
            logger.warning("Document is empty: %s", document.source_path)
            return []

        return [
            Section(
                id=make_section_id(
                    document_id=document.id,
                    ordinal=0,
                    heading=None,
                    text=text,
                ),
                document_id=document.id,
                ordinal=0,
                heading=None,
                level=None,
                text=text,
                metadata={
                    **document.metadata,
                    "section_heading": None,
                    "section_level": None,
                    "section_ordinal": 0,
                },
            )
        ]


class PolicySectioner:
    """Splits policy-like text into numbered sections.

    Recognizes examples like:
    - 1 PURPOSE
    - 1. PURPOSE
    - 3.1 Primary Caregiver
    - 3.1. Primary Caregiver
    """

    _SECTION_RE = re.compile(
        r"^(\d+(?:\.\d+)*\.?)\s+(.+?)\s*$",
        re.MULTILINE,
    )

    def section(self, document: Document, config: ContentTypeConfig) -> list[Section]:
        """Split the document into sections based on policy-like numbering."""
        raw = document.text
        matches = list(self._SECTION_RE.finditer(raw))

        if not matches:
            logger.warning(
                "No policy sections detected; falling back to plaintext sectioning: %s",
                document.source_path,
            )
            return PlaintextSectioner().section(document, config)

        sections: list[Section] = []
        ordinal = 0

        preamble = raw[: matches[0].start()].strip()
        if preamble and config.sectioning.preserve_preamble:
            sections.append(
                self._build_section(
                    document=document,
                    ordinal=ordinal,
                    heading=None,
                    level=None,
                    text=preamble,
                )
            )
            ordinal += 1

        for index, match in enumerate(matches):
            number = match.group(1).rstrip(".")
            title = match.group(2).strip()
            heading = f"{number} {title}"

            body_start = match.end()
            body_end = (
                matches[index + 1].start() if index + 1 < len(matches) else len(raw)
            )
            body = raw[body_start:body_end].strip()

            if not body:
                logger.debug(
                    "Skipping empty policy section: document=%s heading=%s",
                    document.source_path,
                    heading,
                )
                continue

            level = number.count(".") + 1

            sections.append(
                self._build_section(
                    document=document,
                    ordinal=ordinal,
                    heading=heading,
                    level=level,
                    text=f"{heading}\n\n{body}".strip(),
                )
            )
            ordinal += 1

        return sections

    def _build_section(
        self,
        document: Document,
        ordinal: int,
        heading: str | None,
        level: int | None,
        text: str,
    ) -> Section:
        """Helper to build a Section with a stable ID."""
        return Section(
            id=make_section_id(
                document_id=document.id,
                ordinal=ordinal,
                heading=heading,
                text=text,
            ),
            document_id=document.id,
            ordinal=ordinal,
            heading=heading,
            level=level,
            text=text,
            metadata={
                **document.metadata,
                "section_heading": heading,
                "section_level": level,
                "section_ordinal": ordinal,
            },
        )


class MarkdownSectioner:
    """Splits Markdown documents into heading-based sections."""

    def __init__(self) -> None:
        self._parser = MarkdownIt("commonmark")

    def section(self, document: Document, config: ContentTypeConfig) -> list[Section]:
        """Split the document into sections based on markdown headings."""
        target_level = config.sectioning.markdown_heading_level
        target_tag = f"h{target_level}"
        preserve_code = config.sectioning.preserve_code_blocks

        tokens = self._parser.parse(document.text)

        sections: list[Section] = []
        current_heading: str | None = None
        current_level: int | None = None
        current_body: list[str] = []

        index = 0
        while index < len(tokens):
            token = tokens[index]

            if token.type == "heading_open" and token.tag == target_tag:
                if current_heading is not None or current_body:
                    self._append_markdown_section(
                        sections=sections,
                        document=document,
                        heading=current_heading,
                        level=current_level,
                        body="".join(current_body).strip(),
                    )

                inline = tokens[index + 1]
                current_heading = self._render_inline(inline, preserve_code).strip()
                current_level = target_level
                current_body = []
                index += 3
                continue

            rendered = self._render_token(token, preserve_code)
            if rendered:
                current_body.append(rendered)

            index += 1

        if current_heading is not None or current_body:
            self._append_markdown_section(
                sections=sections,
                document=document,
                heading=current_heading,
                level=current_level,
                body="".join(current_body).strip(),
            )

        if not sections:
            logger.warning(
                "No markdown sections detected; falling back to plaintext sectioning: %s",
                document.source_path,
            )
            return PlaintextSectioner().section(document, config)

        return sections

    def _append_markdown_section(
        self,
        sections: list[Section],
        document: Document,
        heading: str | None,
        level: int | None,
        body: str,
    ) -> None:
        """Helper to build and append a Section with a stable ID."""
        text = f"{heading}\n\n{body}".strip() if heading else body.strip()

        if not text:
            return

        ordinal = len(sections)

        sections.append(
            Section(
                id=make_section_id(
                    document_id=document.id,
                    ordinal=ordinal,
                    heading=heading,
                    text=text,
                ),
                document_id=document.id,
                ordinal=ordinal,
                heading=heading,
                level=level,
                text=text,
                metadata={
                    **document.metadata,
                    "section_heading": heading,
                    "section_level": level,
                    "section_ordinal": ordinal,
                },
            )
        )

    def _render_token(self, token: Token, preserve_code: bool) -> str:
        """Render a markdown-it token to text, preserving structure for certain types."""

        if token.type == "heading_open":
            level = int(token.tag[1])
            return "#" * level + " "

        if token.type == "heading_close":
            return "\n\n"

        if token.type == "paragraph_close":
            return "\n\n"

        if token.type == "fence":
            if preserve_code:
                info = token.info or ""
                return f"```{info}\n{token.content}```\n\n"
            return ""

        if token.type == "code_block":
            return f"{token.content}\n\n" if preserve_code else ""

        if token.type == "inline":
            return self._render_inline(token, preserve_code)

        if token.type in ("bullet_list_close", "ordered_list_close", "list_item_close"):
            return "\n"

        return ""

    def _render_inline(self, token: Token, preserve_code: bool) -> str:
        """Render an inline token, preserving code spans if needed."""
        parts: list[str] = []

        for child in token.children or []:
            if child.type == "text":
                parts.append(child.content)
            elif child.type == "code_inline":
                parts.append(f"`{child.content}`" if preserve_code else child.content)
            elif child.type == "softbreak":
                parts.append(" ")
            elif child.type == "hardbreak":
                parts.append("\n")

        return "".join(parts)


def get_sectioner(sectioner_name: str):
    """Factory function to get the appropriate sectioner based on the name."""
    if sectioner_name == "plaintext":
        return PlaintextSectioner()

    if sectioner_name == "policy":
        return PolicySectioner()

    if sectioner_name == "markdown":
        return MarkdownSectioner()

    raise ValueError(f"Unknown sectioner: {sectioner_name}")
