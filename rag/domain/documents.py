from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """Raw document after reading from disk.

    This is the file-level object.
    It does not know about sections or chunks yet.
    """

    id: str
    source_path: str
    content_type: str
    title: str | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Section:
    """Logical section inside a document.

    This is the structure-aware unit:
    - markdown heading section
    - policy numbered section
    - plaintext fallback section
    """

    id: str
    document_id: str
    ordinal: int
    heading: str | None
    level: int | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """Final retrieval unit.

    This is what eventually gets embedded and stored in Qdrant.
    """

    id: str
    document_id: str
    section_id: str
    ordinal: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
