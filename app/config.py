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

    request_timeout: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
