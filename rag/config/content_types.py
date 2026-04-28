from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


ReaderType = Literal["text"]
SectionerType = Literal["markdown", "plaintext", "policy", "faq"]
ChunkerType = Literal["single", "paragraph", "section_window"]


class SectioningConfig(BaseModel):
    markdown_heading_level: int = 2
    preserve_code_blocks: bool = False
    preserve_preamble: bool = True


class ChunkingConfig(BaseModel):
    max_tokens: int
    overlap: int = 0


class ContentTypeConfig(BaseModel):
    source_dir: Path
    file_patterns: list[str]

    reader: ReaderType
    sectioner: SectionerType
    chunker: ChunkerType

    sectioning: SectioningConfig = Field(default_factory=SectioningConfig)
    chunking: ChunkingConfig
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContentTypesConfig(BaseModel):
    content_types: dict[str, ContentTypeConfig]


def load_content_types_config(config_path: Path) -> ContentTypesConfig:
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    return ContentTypesConfig.model_validate(raw_config)
