import hashlib
import re
from pathlib import Path


def stable_hash(value: str, length: int = 12) -> str:
    """Create a short deterministic hash from a string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def slugify(value: str | None, fallback: str = "untitled") -> str:
    """Create a readable ID-safe slug."""
    if not value:
        return fallback

    slug = value.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")

    return slug or fallback


def make_document_id(content_type: str, source_path: Path) -> str:
    """Create a stable document ID based on content type and source path.

    This should remain stable as long as the file path remains stable.
    """
    normalized_path = source_path.as_posix()
    path_hash = stable_hash(normalized_path)

    return f"doc_{content_type}_{source_path.stem}_{path_hash}"


def make_section_id(
    document_id: str,
    ordinal: int,
    heading: str | None,
    text: str,
) -> str:
    """Create a stable section ID.

    We include ordinal because repeated headings are possible.
    We include a short content hash to reduce accidental collisions.
    """
    heading_slug = slugify(heading, fallback="section")
    content_hash = stable_hash(text[:500])

    return f"sec_{document_id}_{ordinal:04d}_{heading_slug}_{content_hash}"


def make_chunk_id(
    section_id: str,
    ordinal: int,
    text: str,
) -> str:
    """Create a stable chunk ID."""
    content_hash = stable_hash(text[:500])

    return f"chk_{section_id}_{ordinal:04d}_{content_hash}"
