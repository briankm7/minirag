"""Application configuration, loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "minirag"

    # Providers: "fake" keeps the whole stack runnable offline (tests, CI, demos).
    embedding_provider: Literal["gemini", "fake"] = "fake"
    generation_provider: Literal["anthropic", "fake"] = "fake"

    gemini_api_key: str | None = None
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    max_output_tokens: int = 1024

    # ":memory:" runs an embedded Qdrant, so no external service is required.
    qdrant_url: str = ":memory:"
    qdrant_api_key: str | None = None
    collection_name: str = "documents"

    chunk_size: int = 400
    chunk_overlap: int = 80
    top_k: int = 4

    # Retrieval strategy. "hybrid" runs dense and BM25 and fuses the rankings;
    # the single-retriever modes exist so the two can be compared on the same
    # corpus without redeploying.
    retrieval_mode: Literal["dense", "lexical", "hybrid"] = "hybrid"
    rrf_k: int = 60
    # Each retriever returns top_k * candidate_multiplier passages before fusion
    # and reranking. Reranking can only reorder what retrieval returned, so this
    # is the knob that trades latency for recall.
    candidate_multiplier: int = 4

    # "lexical" is a cheap offline baseline, not a cross-encoder; see
    # app/core/reranking.py.
    reranker_provider: Literal["cohere", "lexical", "none"] = "lexical"
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-v3.5"

    request_timeout: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
