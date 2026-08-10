from __future__ import annotations

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    """Offline settings: embedded Qdrant, deterministic providers."""
    return Settings(
        embedding_provider="fake",
        generation_provider="fake",
        qdrant_url=":memory:",
        collection_name="test_documents",
        embedding_dimensions=64,
        chunk_size=40,
        chunk_overlap=10,
        top_k=3,
    )


@pytest.fixture
async def client(settings: Settings):
    """An HTTP client bound to a fresh app, with lifespan events honoured."""
    app = create_app(settings)
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
