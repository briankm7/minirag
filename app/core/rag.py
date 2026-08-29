"""The retrieval-augmented generation pipeline.

This module wires chunking, embeddings, dense and lexical retrieval, fusion,
reranking and generation together. It owns no I/O details of its own: every
external dependency arrives through the constructor, which is what makes the
pipeline unit-testable.

Retrieval runs in four stages:

1. **Recall.** Dense search and BM25 each return a candidate pool several times
   larger than the answer size. Anything missed here is lost for good, since no
   later stage can retrieve a passage that was never returned.
2. **Fusion.** Reciprocal Rank Fusion merges the two rankings on rank rather
   than score, because the two score scales are not comparable.
3. **Precision.** The reranker reorders the surviving candidates by scoring each
   one against the query directly.
4. **Grounding.** The top passages become numbered context for the generator,
   which is instructed to answer only from them and to cite them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.core.chunking import chunk_text
from app.core.embeddings import EmbeddingProvider
from app.core.generation import GenerationProvider
from app.core.lexical import LexicalIndex
from app.core.reranking import NoopReranker, Reranker
from app.core.retrieval import (
    DEFAULT_RRF_K,
    RetrievalMode,
    SearchHit,
    reciprocal_rank_fusion,
)
from app.core.vectorstore import VectorStore


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
        lexical: LexicalIndex | None = None,
        reranker: Reranker | None = None,
        mode: RetrievalMode = "hybrid",
        rrf_k: int = DEFAULT_RRF_K,
        candidate_multiplier: int = 4,
    ) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")
        self._store = store
        self._embeddings = embeddings
        self._generator = generator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k
        self._lexical = lexical if lexical is not None else LexicalIndex()
        self._reranker = reranker if reranker is not None else NoopReranker()
        self._mode = mode
        self._rrf_k = rrf_k
        self._candidate_multiplier = candidate_multiplier

    @property
    def lexical(self) -> LexicalIndex:
        return self._lexical

    async def ingest(self, *, title: str, text: str) -> IngestResult:
        """Chunk, embed and store a document in both indexes.

        The chunk ids returned by the vector store are reused as the lexical
        index's keys. That shared identity is what allows fusion to recognise
        the same passage arriving from two different retrievers.

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
        chunk_ids = await self._store.upsert(
            document_id=document_id, title=title, chunks=payloads, vectors=vectors
        )
        for chunk_id, body in zip(chunk_ids, payloads, strict=True):
            self._lexical.add(
                chunk_id=chunk_id, document_id=document_id, title=title, text=body
            )
        return IngestResult(document_id=document_id, chunks=len(chunks))

    async def warm_lexical_index(self) -> int:
        """Rebuild the lexical index from the vector store.

        Called at startup so a restarted process does not serve hybrid queries
        against an empty BM25 index.

        Returns:
            The number of passages indexed.
        """
        await self._store.ensure_collection()
        self._lexical.clear()
        self._lexical.add_many(await self._store.scroll_all())
        return len(self._lexical)

    async def _dense(self, query: str, *, limit: int) -> list[SearchHit]:
        vector = (await self._embeddings.embed([query], task="query"))[0]
        return await self._store.search(vector, limit=limit)

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[SearchHit]:
        """Retrieve passages for ``query`` using the configured strategy.

        Args:
            query: The user's question or search string.
            limit: How many passages to return. Defaults to the configured
                ``top_k``.
            mode: Override the retrieval strategy for this call. Useful for
                comparing strategies against the same corpus without a restart.

        Returns:
            Passages ordered best-first. The ``score`` scale depends on the
            final stage that touched them; see :class:`SearchHit`.
        """
        await self._store.ensure_collection()
        limit = limit or self._top_k
        mode = mode or self._mode
        candidates = limit * self._candidate_multiplier

        rankings: list[list[SearchHit]] = []
        if mode in ("dense", "hybrid"):
            rankings.append(await self._dense(query, limit=candidates))
        if mode in ("lexical", "hybrid"):
            rankings.append(self._lexical.search(query, limit=candidates))

        if len(rankings) == 1:
            pool = rankings[0]
        else:
            pool = reciprocal_rank_fusion(rankings, k=self._rrf_k, limit=candidates)

        return await self._reranker.rerank(query=query, hits=pool, limit=limit)

    async def ask(
        self,
        question: str,
        *,
        limit: int | None = None,
        mode: RetrievalMode | None = None,
    ) -> Answer:
        hits = await self.search(question, limit=limit, mode=mode)
        answer = await self._generator.generate(
            question=question, context=format_context(hits)
        )
        return Answer(answer=answer, sources=hits)
