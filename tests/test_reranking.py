from __future__ import annotations

import httpx
import pytest

from app.core.reranking import (
    CohereReranker,
    LexicalReranker,
    NoopReranker,
    Reranker,
)
from app.core.retrieval import SearchHit

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def hit(chunk_id: str, text: str, score: float = 0.0) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id, document_id="doc", title="Title", text=text, score=score
    )


def test_providers_satisfy_the_protocol():
    assert isinstance(NoopReranker(), Reranker)
    assert isinstance(LexicalReranker(), Reranker)


async def test_noop_preserves_order_and_truncates():
    hits = [hit("a", "one"), hit("b", "two"), hit("c", "three")]
    result = await NoopReranker().rerank(query="anything", hits=hits, limit=2)
    assert [h.chunk_id for h in result] == ["a", "b"]


async def test_lexical_reranker_promotes_better_coverage():
    """The reranker must be able to overturn the retrieval order, or it is inert."""
    hits = [
        hit("weak", "a passage about unrelated matters"),
        hit("strong", "cosine similarity in vector search"),
    ]
    result = await LexicalReranker().rerank(
        query="cosine similarity vector", hits=hits, limit=2
    )
    assert [h.chunk_id for h in result] == ["strong", "weak"]


async def test_density_separates_equal_coverage():
    hits = [
        hit("diluted", "qdrant " + " ".join(f"filler{n}" for n in range(100))),
        hit("focused", "qdrant vector database"),
    ]
    result = await LexicalReranker().rerank(query="qdrant", hits=hits, limit=2)
    assert result[0].chunk_id == "focused"


async def test_reranker_replaces_the_incoming_score():
    hits = [hit("a", "cosine similarity", score=999.0)]
    result = await LexicalReranker().rerank(query="cosine", hits=hits, limit=1)
    assert result[0].score < 2.0


async def test_empty_query_or_passage_scores_zero():
    reranker = LexicalReranker()
    assert reranker.score("", "some text") == 0.0
    assert reranker.score("query", "") == 0.0


async def test_reranking_nothing_returns_nothing():
    assert await LexicalReranker().rerank(query="q", hits=[], limit=3) == []
    assert await CohereReranker("key").rerank(query="q", hits=[], limit=3) == []


def test_cohere_requires_an_api_key():
    with pytest.raises(ValueError):
        CohereReranker("")


async def test_cohere_maps_returned_indices_back_to_hits(monkeypatch):
    """The API answers with positions, so the mapping back must stay positional."""
    hits = [hit("a", "first"), hit("b", "second"), hit("c", "third")]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.42},
                    {"index": 99, "relevance_score": 0.10},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)

    result = await CohereReranker("key").rerank(query="q", hits=hits, limit=3)
    assert [h.chunk_id for h in result] == ["c", "a"]
    assert result[0].score == pytest.approx(0.91)
