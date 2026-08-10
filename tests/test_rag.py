from __future__ import annotations

import math

import pytest

from app.core.embeddings import FakeEmbeddings
from app.core.generation import FakeGenerator
from app.core.rag import RagService, format_context
from app.core.vectorstore import SearchHit, VectorStore

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_embeddings_are_deterministic_and_normalised():
    embedder = FakeEmbeddings(dimensions=32)
    first = await embedder.embed(["hello world"], task="document")
    second = await embedder.embed(["hello world"], task="document")

    assert first == second
    assert len(first[0]) == 32
    assert math.isclose(math.sqrt(sum(v * v for v in first[0])), 1.0, rel_tol=1e-6)


async def test_different_text_produces_different_vectors():
    embedder = FakeEmbeddings(dimensions=32)
    vectors = await embedder.embed(["alpha", "beta"], task="document")
    assert vectors[0] != vectors[1]


def test_context_is_numbered_for_citation():
    hits = [
        SearchHit(chunk_id="1", document_id="d", title="A", text="first", score=0.9),
        SearchHit(chunk_id="2", document_id="d", title="B", text="second", score=0.8),
    ]
    context = format_context(hits)
    assert context.startswith("[1] (A) first")
    assert "[2] (B) second" in context


async def test_generator_refuses_without_context():
    answer = await FakeGenerator().generate(question="anything?", context="")
    assert "cannot answer" in answer.lower()


@pytest.fixture
async def service() -> RagService:
    store = VectorStore.from_url(":memory:", collection="unit", dimensions=64)
    await store.ensure_collection()
    return RagService(
        store=store,
        embeddings=FakeEmbeddings(dimensions=64),
        generator=FakeGenerator(),
        chunk_size=40,
        chunk_overlap=10,
        top_k=3,
    )


async def test_ingest_rejects_empty_documents(service: RagService):
    with pytest.raises(ValueError):
        await service.ingest(title="empty", text="   ")


async def test_ingested_document_is_retrievable(service: RagService):
    text = " ".join(f"sentence{n}" for n in range(120))
    result = await service.ingest(title="Doc", text=text)
    assert result.chunks > 1

    hits = await service.search("sentence0", limit=3)
    assert hits
    assert all(hit.document_id == result.document_id for hit in hits)
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


async def test_ask_returns_answer_with_sources(service: RagService):
    await service.ingest(title="Doc", text="Qdrant stores dense vectors for search.")
    answer = await service.ask("What does Qdrant store?")

    assert answer.answer
    assert answer.sources
    assert answer.sources[0].title == "Doc"
