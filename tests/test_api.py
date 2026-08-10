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
