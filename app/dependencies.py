"""Composition root.

Providers are chosen here and nowhere else, so switching from the offline
"fake" stack to real vendors never touches business logic.
"""

from __future__ import annotations

from app.config import Settings
from app.core.embeddings import EmbeddingProvider, FakeEmbeddings, GeminiEmbeddings
from app.core.generation import AnthropicGenerator, FakeGenerator, GenerationProvider
from app.core.rag import RagService
from app.core.vectorstore import VectorStore


def build_embeddings(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "gemini":
        return GeminiEmbeddings(
            settings.gemini_api_key or "",
            model=settings.gemini_embedding_model,
            dimensions=settings.embedding_dimensions,
            timeout=settings.request_timeout,
        )
    return FakeEmbeddings(dimensions=settings.embedding_dimensions)


def build_generator(settings: Settings) -> GenerationProvider:
    if settings.generation_provider == "anthropic":
        return AnthropicGenerator(
            settings.anthropic_api_key or "",
            model=settings.anthropic_model,
            max_tokens=settings.max_output_tokens,
            timeout=settings.request_timeout,
        )
    return FakeGenerator()


def build_store(settings: Settings) -> VectorStore:
    return VectorStore.from_url(
        settings.qdrant_url,
        collection=settings.collection_name,
        dimensions=settings.embedding_dimensions,
        api_key=settings.qdrant_api_key,
    )


def build_service(settings: Settings, store: VectorStore) -> RagService:
    return RagService(
        store=store,
        embeddings=build_embeddings(settings),
        generator=build_generator(settings),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
    )
