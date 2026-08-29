"""The case hybrid retrieval exists to solve.

Dense retrieval matches meaning, so it is weakest where meaning is not the
point: exact identifiers, codes and reference numbers. These tests make that
failure explicit rather than probable, using an embedding stub whose behaviour
is stated up front — the point is to test the pipeline's wiring, not to
rediscover that a hash-based stub ranks arbitrarily.
"""

from __future__ import annotations

import pytest

from app.core.generation import FakeGenerator
from app.core.rag import RagService
from app.core.reranking import LexicalReranker, NoopReranker
from app.core.vectorstore import VectorStore

pytestmark = pytest.mark.anyio

REFERENCE = "ES-2024-01847"
QUERY = f"billing reference {REFERENCE}"


class TopicEmbeddings:
    """Embeds on topic alone, ignoring identifiers.

    Anything mentioning billing lands on one axis; everything else lands on
    another. This is a caricature of a real bi-encoder, but it is a faithful one
    in the respect that matters here: an exact reference number contributes
    nothing to the vector.
    """

    dimensions = 4

    async def embed(self, texts: list[str], *, task: str = "document") -> list[list[float]]:
        return [
            [1.0, 0.0, 0.0, 0.0] if "billing" in text.lower() else [0.0, 1.0, 0.0, 0.0]
            for text in texts
        ]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def build_service(reranker=None, mode="hybrid") -> RagService:
    store = VectorStore.from_url(":memory:", collection="hybrid", dimensions=4)
    await store.ensure_collection()
    service = RagService(
        store=store,
        embeddings=TopicEmbeddings(),
        generator=FakeGenerator(),
        chunk_size=60,
        chunk_overlap=10,
        top_k=3,
        reranker=reranker or NoopReranker(),
        mode=mode,
        candidate_multiplier=4,
    )
    # More on-topic decoys than the candidate pool holds (top_k * multiplier =
    # 12), so a passage dense search ranks below them is genuinely out of reach
    # of anything downstream — not merely ranked low.
    for n in range(20):
        await service.ingest(
            title=f"Billing note {n}",
            text=f"General billing guidance number {n} about invoices and payment terms.",
        )
    await service.ingest(
        title="Refund record",
        text=f"Refund receipt {REFERENCE} was issued to the customer last quarter.",
    )
    return service


def texts(hits) -> str:
    return " ".join(hit.text for hit in hits)


async def test_dense_search_misses_the_exact_reference():
    service = await build_service(mode="dense")
    hits = await service.search(QUERY, limit=3)

    assert hits, "dense search should still return on-topic passages"
    assert REFERENCE not in texts(hits)


async def test_lexical_search_finds_the_exact_reference():
    service = await build_service(mode="lexical")
    hits = await service.search(QUERY, limit=3)
    assert hits[0].text.count(REFERENCE) == 1


async def test_hybrid_recovers_what_dense_lost():
    """The default configuration: hybrid retrieval plus the lexical reranker."""
    service = await build_service(reranker=LexicalReranker(), mode="hybrid")
    hits = await service.search(QUERY, limit=3)
    assert REFERENCE in texts(hits)


async def test_reranking_promotes_the_exact_match_to_first():
    """Fusion restores recall; reranking is what fixes the ordering."""
    service = await build_service(reranker=LexicalReranker(), mode="hybrid")
    hits = await service.search(QUERY, limit=3)
    assert REFERENCE in hits[0].text


async def test_reranker_cannot_recover_a_passage_retrieval_never_returned():
    """The limit of reranking, stated as a test so it is not forgotten."""
    service = await build_service(reranker=LexicalReranker(), mode="dense")
    hits = await service.search(QUERY, limit=3)
    assert REFERENCE not in texts(hits)


async def test_ask_is_grounded_in_hybrid_results():
    service = await build_service(reranker=LexicalReranker(), mode="hybrid")
    answer = await service.ask(QUERY)
    assert answer.answer
    assert REFERENCE in answer.sources[0].text


async def test_fusion_alone_does_not_guarantee_the_top_slot():
    """Worth stating plainly, because it is easy to assume otherwise.

    Fusion rewards agreement between retrievers. The decoys are returned by
    both, so they accumulate two contributions each, while the exact-match
    passage is found by BM25 alone and accumulates one. Fusion therefore gets
    the passage into the candidate pool but not to the top of it — which is
    precisely the job the reranker does, and why the default pipeline has one.
    """
    service = await build_service(reranker=NoopReranker(), mode="hybrid")
    hits = await service.search(QUERY, limit=3)
    assert REFERENCE not in texts(hits)


async def test_mode_can_be_overridden_per_call():
    service = await build_service(reranker=LexicalReranker(), mode="dense")
    assert REFERENCE not in texts(await service.search(QUERY, limit=3))
    assert REFERENCE in texts(await service.search(QUERY, limit=3, mode="hybrid"))


async def test_lexical_index_is_rebuilt_from_the_vector_store():
    """A restart must not silently degrade hybrid search to dense-only."""
    service = await build_service(reranker=LexicalReranker(), mode="hybrid")
    expected = len(service.lexical)
    assert expected > 0

    service.lexical.clear()
    assert len(service.lexical) == 0

    restored = await service.warm_lexical_index()
    assert restored == expected
    assert REFERENCE in texts(await service.search(QUERY, limit=3))


async def test_candidate_multiplier_must_be_at_least_one():
    store = VectorStore.from_url(":memory:", collection="invalid", dimensions=4)
    with pytest.raises(ValueError):
        RagService(
            store=store,
            embeddings=TopicEmbeddings(),
            generator=FakeGenerator(),
            chunk_size=60,
            chunk_overlap=10,
            top_k=3,
            candidate_multiplier=0,
        )
