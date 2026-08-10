"""FastAPI application exposing the RAG pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status

from app.config import Settings, get_settings
from app.core.rag import RagService
from app.dependencies import build_service, build_store
from app.schemas import (
    AskIn,
    AskOut,
    DocumentIn,
    DocumentOut,
    HealthOut,
    SearchIn,
    SearchOut,
    SourceOut,
)

DESCRIPTION = """
A compact retrieval-augmented generation service: ingest documents, search them
semantically, and get answers grounded in the retrieved passages.

External providers sit behind interfaces, so the whole API runs offline with
deterministic stand-ins when no API keys are configured.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Own the vector store connection for the lifetime of the process."""
    settings: Settings = app.state.settings
    store = build_store(settings)
    await store.ensure_collection()
    app.state.store = store
    app.state.service = build_service(settings, store)
    try:
        yield
    finally:
        await store.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="minirag",
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings or get_settings()
    register_routes(app)
    return app


def get_service(request: Request) -> RagService:
    return request.app.state.service


ServiceDep = Annotated[RagService, Depends(get_service)]


def _to_sources(hits) -> list[SourceOut]:
    return [
        SourceOut(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            title=hit.title,
            text=hit.text,
            score=hit.score,
        )
        for hit in hits
    ]


def register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthOut, tags=["system"])
    async def health(request: Request) -> HealthOut:
        settings: Settings = request.app.state.settings
        return HealthOut(
            status="ok",
            embedding_provider=settings.embedding_provider,
            generation_provider=settings.generation_provider,
            indexed_chunks=await request.app.state.store.count(),
        )

    @app.post(
        "/documents",
        response_model=DocumentOut,
        status_code=status.HTTP_201_CREATED,
        tags=["documents"],
    )
    async def ingest(
        payload: DocumentIn, service: ServiceDep
    ) -> DocumentOut:
        try:
            result = await service.ingest(title=payload.title, text=payload.text)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        return DocumentOut(document_id=result.document_id, chunks=result.chunks)

    @app.post("/search", response_model=SearchOut, tags=["retrieval"])
    async def search(
        payload: SearchIn, service: ServiceDep
    ) -> SearchOut:
        hits = await service.search(payload.query, limit=payload.limit)
        return SearchOut(query=payload.query, results=_to_sources(hits))

    @app.post("/ask", response_model=AskOut, tags=["retrieval"])
    async def ask(payload: AskIn, service: ServiceDep) -> AskOut:
        result = await service.ask(payload.query, limit=payload.limit)
        return AskOut(
            question=payload.query,
            answer=result.answer,
            sources=_to_sources(result.sources),
        )


app = create_app()
