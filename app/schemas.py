"""Request and response models for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.retrieval import RetrievalMode


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=200, examples=["Qdrant overview"])
    text: str = Field(min_length=1, examples=["Qdrant is a vector database..."])


class DocumentOut(BaseModel):
    document_id: str
    chunks: int


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int | None = Field(default=None, ge=1, le=20)
    mode: RetrievalMode | None = Field(
        default=None,
        description=(
            "Override the retrieval strategy for this request. 'dense' is "
            "vector search only, 'lexical' is BM25 only, 'hybrid' fuses both. "
            "Defaults to the server configuration."
        ),
    )


class SourceOut(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    text: str
    score: float = Field(
        description=(
            "Relevance within this response only. The scale depends on the "
            "last stage applied (cosine, BM25, fused rank or reranker score), "
            "so scores are not comparable across retrieval modes."
        )
    )


class SearchOut(BaseModel):
    query: str
    mode: RetrievalMode
    results: list[SourceOut]


class AskIn(SearchIn):
    pass


class AskOut(BaseModel):
    question: str
    answer: str
    mode: RetrievalMode
    sources: list[SourceOut]


class HealthOut(BaseModel):
    status: str
    embedding_provider: str
    generation_provider: str
    reranker_provider: str
    retrieval_mode: RetrievalMode
    indexed_chunks: int
    lexical_passages: int
