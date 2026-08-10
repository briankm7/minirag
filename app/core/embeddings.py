"""Embedding providers.

The rest of the application depends on the :class:`EmbeddingProvider` protocol
rather than on a concrete vendor. That keeps the retrieval pipeline testable
offline and makes swapping providers a configuration change.
"""

from __future__ import annotations

import hashlib
import math
from typing import Literal, Protocol, runtime_checkable

import httpx

TaskType = Literal["document", "query"]

_GEMINI_TASK = {
    "document": "RETRIEVAL_DOCUMENT",
    "query": "RETRIEVAL_QUERY",
}


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into dense vectors."""

    dimensions: int

    async def embed(self, texts: list[str], *, task: TaskType) -> list[list[float]]:
        ...


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


class FakeEmbeddings:
    """Deterministic, dependency-free embeddings.

    Vectors are derived from a hash of the text, so identical text always maps
    to an identical vector and similar-but-not-equal text maps elsewhere. This
    is meaningless semantically but perfectly adequate for exercising the
    pipeline in tests and CI, where no API key is available.
    """

    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions

    async def embed(self, texts: list[str], *, task: TaskType = "document") -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            seed = hashlib.sha256(text.strip().lower().encode()).digest()
            raw = [
                (seed[i % len(seed)] ^ (i * 31 % 251)) / 255.0 - 0.5
                for i in range(self.dimensions)
            ]
            vectors.append(_l2_normalize(raw))
        return vectors


class GeminiEmbeddings:
    """Google Gemini embeddings via the REST API.

    Uses asymmetric task types: documents and queries are embedded with
    different objectives, which measurably improves retrieval over embedding
    both sides identically.
    """

    _ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("A Gemini API key is required")
        self._api_key = api_key
        self._model = model
        self.dimensions = dimensions
        self._timeout = timeout

    async def embed(self, texts: list[str], *, task: TaskType = "document") -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "requests": [
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": _GEMINI_TASK[task],
                    "outputDimensionality": self.dimensions,
                }
                for text in texts
            ]
        }
        url = self._ENDPOINT.format(model=self._model)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                url, json=payload, headers={"x-goog-api-key": self._api_key}
            )
            response.raise_for_status()
            data = response.json()
        return [_l2_normalize(item["values"]) for item in data["embeddings"]]
