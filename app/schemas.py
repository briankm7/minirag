"""Request and response models for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=200, examples=["Qdrant overview"])
    text: str = Field(min_length=1, examples=["Qdrant is a vector database..."])


class DocumentOut(BaseModel):
    document_id: str
    chunks: int


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    text: str
    score: float


class SearchOut(BaseModel):
    query: str
    results: list[SourceOut]


class AskIn(SearchIn):
    pass


class AskOut(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]


class HealthOut(BaseModel):
    status: str
    embedding_provider: str
    generation_provider: str
    indexed_chunks: int
