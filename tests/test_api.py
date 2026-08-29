from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_health_reports_configuration(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["embedding_provider"] == "fake"
    assert body["indexed_chunks"] == 0


async def test_ingest_search_and_ask_end_to_end(client: AsyncClient):
    ingest = await client.post(
        "/documents",
        json={
            "title": "Vector databases",
            "text": (
                "Qdrant is an open source vector database. "
                "It stores dense embeddings and supports cosine similarity search. "
                + " ".join(f"filler{n}" for n in range(200))
            ),
        },
    )
    assert ingest.status_code == 201
    document_id = ingest.json()["document_id"]
    assert ingest.json()["chunks"] >= 1

    health = await client.get("/health")
    assert health.json()["indexed_chunks"] > 0

    search = await client.post("/search", json={"query": "filler10", "limit": 2})
    assert search.status_code == 200
    results = search.json()["results"]
    assert 0 < len(results) <= 2
    assert results[0]["document_id"] == document_id

    ask = await client.post("/ask", json={"query": "filler10"})
    assert ask.status_code == 200
    body = ask.json()
    assert body["question"] == "filler10"
    assert body["answer"]
    assert body["sources"]


async def test_empty_document_is_rejected(client: AsyncClient):
    response = await client.post("/documents", json={"title": "Empty", "text": "   "})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "", "text": "content"},
        {"title": "Title"},
        {"text": "content"},
    ],
)
async def test_invalid_payloads_are_rejected(client: AsyncClient, payload: dict):
    response = await client.post("/documents", json=payload)
    assert response.status_code == 422


async def test_search_limit_is_bounded(client: AsyncClient):
    response = await client.post("/search", json={"query": "anything", "limit": 999})
    assert response.status_code == 422


async def test_health_reports_the_retrieval_configuration(client: AsyncClient):
    body = (await client.get("/health")).json()
    assert body["retrieval_mode"] == "hybrid"
    assert body["reranker_provider"] == "lexical"
    assert body["lexical_passages"] == 0


async def test_both_indexes_grow_on_ingest(client: AsyncClient):
    await client.post(
        "/documents",
        json={"title": "Refunds", "text": "Refund receipt ES-2024-01847 was issued."},
    )
    body = (await client.get("/health")).json()
    assert body["indexed_chunks"] > 0
    assert body["lexical_passages"] == body["indexed_chunks"]


async def test_exact_reference_is_retrievable_end_to_end(client: AsyncClient):
    await client.post(
        "/documents",
        json={
            "title": "Billing guidance",
            "text": "General billing guidance about invoices and payment terms. "
            + " ".join(f"filler{n}" for n in range(300)),
        },
    )
    await client.post(
        "/documents",
        json={"title": "Refunds", "text": "Refund receipt ES-2024-01847 was issued."},
    )

    response = await client.post("/search", json={"query": "ES-2024-01847", "limit": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "hybrid"
    assert "ES-2024-01847" in body["results"][0]["text"]


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
async def test_search_mode_is_echoed_back(client: AsyncClient, mode: str):
    await client.post("/documents", json={"title": "Doc", "text": "vector search basics"})
    response = await client.post("/search", json={"query": "vector", "mode": mode})
    assert response.status_code == 200
    assert response.json()["mode"] == mode


async def test_ask_reports_the_mode_it_used(client: AsyncClient):
    await client.post("/documents", json={"title": "Doc", "text": "vector search basics"})
    body = (await client.post("/ask", json={"query": "vector", "mode": "lexical"})).json()
    assert body["mode"] == "lexical"
    assert body["sources"]


async def test_unknown_retrieval_mode_is_rejected(client: AsyncClient):
    response = await client.post("/search", json={"query": "x", "mode": "magic"})
    assert response.status_code == 422
