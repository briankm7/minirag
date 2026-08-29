"""The composition root is the only place that knows which vendor is in use.

These tests pin that down, because a silent fallback to the offline stand-ins in
a production deployment would look like working software while quietly serving
worse answers.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.core.embeddings import FakeEmbeddings, GeminiEmbeddings
from app.core.generation import AnthropicGenerator, FakeGenerator
from app.core.reranking import CohereReranker, LexicalReranker, NoopReranker
from app.dependencies import (
    build_embeddings,
    build_generator,
    build_reranker,
    build_service,
    build_store,
)


def settings(**overrides) -> Settings:
    base = {
        "embedding_provider": "fake",
        "generation_provider": "fake",
        "reranker_provider": "lexical",
        "qdrant_url": ":memory:",
    }
    return Settings(**{**base, **overrides})


def test_defaults_are_the_offline_stack():
    assert isinstance(build_embeddings(settings()), FakeEmbeddings)
    assert isinstance(build_generator(settings()), FakeGenerator)


@pytest.mark.parametrize(
    ("provider", "expected"),
    [("lexical", LexicalReranker), ("none", NoopReranker)],
)
def test_reranker_provider_selects_the_implementation(provider: str, expected: type):
    assert isinstance(build_reranker(settings(reranker_provider=provider)), expected)


def test_cohere_reranker_is_built_when_configured():
    reranker = build_reranker(
        settings(reranker_provider="cohere", cohere_api_key="key")
    )
    assert isinstance(reranker, CohereReranker)


def test_real_providers_are_built_when_configured():
    assert isinstance(
        build_embeddings(settings(embedding_provider="gemini", gemini_api_key="key")),
        GeminiEmbeddings,
    )
    assert isinstance(
        build_generator(
            settings(generation_provider="anthropic", anthropic_api_key="key")
        ),
        AnthropicGenerator,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding_provider": "gemini"},
        {"generation_provider": "anthropic"},
        {"reranker_provider": "cohere"},
    ],
)
def test_a_missing_api_key_fails_loudly(overrides: dict):
    """Better to refuse to start than to fall back to a stand-in unnoticed."""
    with pytest.raises(ValueError):
        config = settings(**overrides)
        build_embeddings(config)
        build_generator(config)
        build_reranker(config)


def test_service_is_wired_from_settings():
    config = settings(retrieval_mode="lexical", top_k=7, candidate_multiplier=2)
    service = build_service(config, build_store(config))
    assert service.lexical is not None


def test_unknown_provider_values_are_rejected_by_config():
    with pytest.raises(ValueError):
        settings(reranker_provider="nonsense")
