"""Reranking providers.

Retrieval and ranking are different problems. A bi-encoder embeds the query and
the passage separately, so the two never see each other; that independence is
what makes the index precomputable and the search fast, and it is also what
caps its precision. A reranker gives up that independence: it scores query and
passage together, which is far more accurate and far too slow to run over a
whole corpus.

So the pipeline uses each where it is strong. Retrieval casts a wide net over
everything, cheaply. Reranking reorders only the handful of candidates that
survived, expensively. The candidate pool is deliberately larger than the final
answer size, because a reranker can only fix ordering, never recover a passage
that retrieval failed to return.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx

from app.core.lexical import tokenize
from app.core.retrieval import SearchHit


@runtime_checkable
class Reranker(Protocol):
    """Reorders retrieved passages by relevance to the query."""

    async def rerank(
        self, *, query: str, hits: list[SearchHit], limit: int
    ) -> list[SearchHit]:
        ...


class NoopReranker:
    """Keeps the retrieval order untouched.

    Useful for measuring what reranking actually buys: run the same queries with
    this and with a real reranker and compare.
    """

    async def rerank(
        self, *, query: str, hits: list[SearchHit], limit: int
    ) -> list[SearchHit]:
        return hits[:limit]


class LexicalReranker:
    """A cheap, dependency-free reranker scoring query-term coverage.

    This is a baseline, not a cross-encoder, and it is worth being precise about
    the difference: it rewards a passage for containing more of the query's
    distinct terms, and mildly prefers shorter passages at equal coverage. It
    understands no synonyms and no word order.

    It exists so the default configuration reranks with something honest rather
    than with a stub, and so the pipeline's shape can be exercised in tests and
    CI with no API key. Where reranking quality matters, configure a real
    cross-encoder instead.
    """

    def __init__(self, *, length_penalty: float = 0.15) -> None:
        self._length_penalty = length_penalty

    def score(self, query: str, text: str) -> float:
        query_terms = set(tokenize(query))
        if not query_terms:
            return 0.0
        passage_terms = tokenize(text)
        if not passage_terms:
            return 0.0
        matched = query_terms & set(passage_terms)
        coverage = len(matched) / len(query_terms)
        # Density separates passages with equal coverage: the same terms in a
        # shorter passage are more likely to be what the passage is about.
        density = len([t for t in passage_terms if t in matched]) / len(passage_terms)
        return coverage + self._length_penalty * density

    async def rerank(
        self, *, query: str, hits: list[SearchHit], limit: int
    ) -> list[SearchHit]:
        from dataclasses import replace

        rescored = [replace(hit, score=self.score(query, hit.text)) for hit in hits]
        rescored.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return rescored[:limit]


class CohereReranker:
    """Cross-encoder reranking through the Cohere Rerank API.

    The API returns indices into the documents that were sent, so the mapping
    back to the original hits is positional and must not be reordered before the
    response is applied.
    """

    _ENDPOINT = "https://api.cohere.com/v2/rerank"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "rerank-v3.5",
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("A Cohere API key is required")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    async def rerank(
        self, *, query: str, hits: list[SearchHit], limit: int
    ) -> list[SearchHit]:
        from dataclasses import replace

        if not hits:
            return []
        payload = {
            "model": self._model,
            "query": query,
            "documents": [hit.text for hit in hits],
            "top_n": min(limit, len(hits)),
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self._ENDPOINT, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        reranked: list[SearchHit] = []
        for item in data.get("results", []):
            index = item.get("index")
            if index is None or not 0 <= index < len(hits):
                continue
            reranked.append(
                replace(hits[index], score=float(item.get("relevance_score", 0.0)))
            )
        return reranked[:limit]
