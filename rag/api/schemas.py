from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, Any] = Field(default_factory=dict)


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
