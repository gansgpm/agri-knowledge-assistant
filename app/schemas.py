"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

SourceType = Literal["pdf", "url"]
IngestStatus = Literal["ingested", "skipped_duplicate", "failed"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    index_status: Literal["ok", "empty", "model_mismatch"] = Field(
        description="model_mismatch = the index was built with a different embedding model and is locked."
    )
    index_message: str | None = None
    index_chunks: int
    sources: int
    embedding_cache_vectors: int = Field(0, description="Embeddings stored in the on-disk cache.")
    llm_provider: str
    embedding_provider: str
    guardrails: list[str] = Field(default_factory=list, description="Guardrails currently switched on.")


class UrlIngestRequest(BaseModel):
    urls: list[HttpUrl] = Field(..., min_length=1, max_length=20, description="Web pages to ingest.")

    @field_validator("urls")
    @classmethod
    def _only_http(cls, urls: list[HttpUrl]) -> list[HttpUrl]:
        for url in urls:
            if url.scheme not in ("http", "https"):
                raise ValueError("Only http and https URLs are allowed")
        return urls


class IngestItemResult(BaseModel):
    source: str = Field(description="File name or URL.")
    source_type: SourceType
    status: IngestStatus
    chunks_added: int = 0
    pages: int | None = None
    message: str


class IngestResponse(BaseModel):
    results: list[IngestItemResult]
    total_chunks_added: int
    index_chunks: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20, description="Override TOP_K.")

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Question must contain at least 3 non-space characters")
        return value


class Citation(BaseModel):
    ref: int = Field(description="Number used in the answer, e.g. [1].")
    source: str
    source_type: SourceType
    title: str | None = None
    page: int | None = None
    snippet: str
    distance: float = Field(description="L2 distance; lower means more similar.")


class QueryResponse(BaseModel):
    question: str
    answer: str
    answerable: bool = Field(description="False when the sources did not contain enough information.")
    citations: list[Citation]
    guardrail: Literal["prompt_injection", "off_topic", "no_citation"] | None = Field(
        default=None, description="Set when a guardrail blocked or withheld the answer."
    )
    warnings: list[str] = Field(default_factory=list, description="Output-guardrail findings.")
    notes: list[str] = Field(default_factory=list, description="Safety notes added to the answer.")


class SourceInfo(BaseModel):
    source_id: str
    source_type: SourceType
    display_name: str
    chunks: int
    pages: int | None = None
    ingested_at: str


class SourcesResponse(BaseModel):
    sources: list[SourceInfo]
