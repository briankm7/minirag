"""The retrieval-augmented generation pipeline.

This module wires chunking, embeddings, vector search and generation together.
It owns no I/O details of its own: every external dependency arrives through
the constructor, which is what makes the pipeline unit-testable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.core.chunking import chunk_text
from app.core.embeddings import EmbeddingProvider
from app.core.generation import GenerationProvider
from app.core.vectorstore import SearchHit, VectorStore


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    chunks: int


@dataclass(frozen=True)
class Answer:
    answer: str
    sources: list[SearchHit]


def format_context(hits: list[SearchHit]) -> str:
    """Render hits as numbered passages the model can cite."""
    return "\n\n".join(
        f"[{position}] ({hit.title}) {hit.text}" for position, hit in enumerate(hits, start=1)
    )


class RagService:
    def __init__(
        self,
        *,
        store: VectorStore,
        embeddings: EmbeddingProvider,
        generator: GenerationProvider,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._generator = generator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k

    async def ingest(self, *, title: str, text: str) -> IngestResult:
        """Chunk, embed and store a document.

        Raises:
            ValueError: If the text contains no usable content.
        """
        chunks = chunk_text(text, chunk_size=self._chunk_size, overlap=self._chunk_overlap)
        if not chunks:
            raise ValueError("Document contains no indexable text")

        document_id = str(uuid.uuid4())
        payloads = [chunk.text for chunk in chunks]
        vectors = await self._embeddings.embed(payloads, task="document")
        await self._store.ensure_collection()
        await self._store.upsert(
            document_id=document_id, title=title, chunks=payloads, vectors=vectors
        )
        return IngestResult(document_id=document_id, chunks=len(chunks))

    async def search(self, query: str, *, limit: int | None = None) -> list[SearchHit]:
        await self._store.ensure_collection()
        vector = (await self._embeddings.embed([query], task="query"))[0]
        return await self._store.search(vector, limit=limit or self._top_k)

    async def ask(self, question: str, *, limit: int | None = None) -> Answer:
        hits = await self.search(question, limit=limit)
        answer = await self._generator.generate(
            question=question, context=format_context(hits)
        )
        return Answer(answer=answer, sources=hits)
