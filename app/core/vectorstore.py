"""Qdrant-backed vector storage.

Runs against Qdrant Cloud, a local container, or an embedded in-process
instance (``:memory:``) so the project can be cloned and run with no
infrastructure.
"""

from __future__ import annotations

import uuid

from qdrant_client import AsyncQdrantClient, models

from app.core.retrieval import SearchHit

__all__ = ["SearchHit", "VectorStore"]


class VectorStore:
    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        collection: str,
        dimensions: int,
    ) -> None:
        self._client = client
        self._collection = collection
        self._dimensions = dimensions

    @classmethod
    def from_url(cls, url: str, *, collection: str, dimensions: int, api_key: str | None = None):
        if url == ":memory:":
            client = AsyncQdrantClient(location=":memory:")
        else:
            client = AsyncQdrantClient(url=url, api_key=api_key)
        return cls(client, collection=collection, dimensions=dimensions)

    async def ensure_collection(self) -> None:
        """Create the collection if it does not already exist (idempotent)."""
        if await self._client.collection_exists(self._collection):
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=self._dimensions,
                distance=models.Distance.COSINE,
            ),
        )

    async def upsert(
        self,
        *,
        document_id: str,
        title: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> list[str]:
        points = []
        ids: list[str] = []
        for position, (text, vector) in enumerate(zip(chunks, vectors, strict=True)):
            point_id = str(uuid.uuid4())
            ids.append(point_id)
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "document_id": document_id,
                        "title": title,
                        "text": text,
                        "position": position,
                    },
                )
            )
        if points:
            await self._client.upsert(collection_name=self._collection, points=points)
        return ids

    async def search(self, vector: list[float], *, limit: int) -> list[SearchHit]:
        results = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for point in results.points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    chunk_id=str(point.id),
                    document_id=payload.get("document_id", ""),
                    title=payload.get("title", ""),
                    text=payload.get("text", ""),
                    score=point.score,
                )
            )
        return hits

    async def scroll_all(self, *, batch_size: int = 256) -> list[SearchHit]:
        """Read every stored chunk, in pages.

        This exists so the lexical index can be rebuilt at startup from the
        vector store, which is the durable copy. Without it the two retrievers
        would disagree after any restart: Qdrant would still hold the corpus
        while the in-memory BM25 index came up empty, and hybrid search would
        silently degrade to dense-only with no error to notice.

        Scores are zero because nothing has been queried; only the payloads
        matter here.
        """
        hits: list[SearchHit] = []
        offset = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self._collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                hits.append(
                    SearchHit(
                        chunk_id=str(point.id),
                        document_id=payload.get("document_id", ""),
                        title=payload.get("title", ""),
                        text=payload.get("text", ""),
                        score=0.0,
                    )
                )
            if offset is None:
                return hits

    async def delete_document(self, document_id: str) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    async def count(self) -> int:
        result = await self._client.count(collection_name=self._collection)
        return result.count

    async def close(self) -> None:
        await self._client.close()
