from typing import Any

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="User question to answer using retrieved context.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of chunks to retrieve.",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata filters applied during retrieval.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("Query must not be blank.")

        return normalized


class RetrievedChunkResponse(BaseModel):
    rank: int
    score: float | None = None
    source: str | None = None
    source_path: str | None = None
    content_type: str | None = None
    domain: str | None = None
    doc_role: str | None = None
    chunk_id: str | None = None
    document_id: str | None = None
    section_id: str | None = None
    text: str | None = None


class AskResponse(BaseModel):
    run_id: str
    query: str
    filters: dict[str, Any]
    top_k: int
    retrieval_mode: str
    answer: str
    retrieved_chunks: list[RetrievedChunkResponse]
