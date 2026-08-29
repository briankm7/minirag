"""Retrieval contracts and rank fusion.

This module owns the vocabulary shared by every retriever: what a hit is, which
retrieval strategies exist, and how two ranked lists are combined into one.

It deliberately depends on nothing else in the application. Both the dense
retriever (Qdrant) and the lexical retriever (BM25) import from here, which is
what lets them be combined without either knowing the other exists.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

RetrievalMode = Literal["dense", "lexical", "hybrid"]

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class SearchHit:
    """A single retrieved passage.

    ``score`` is only meaningful relative to other hits from the same call. Its
    scale depends on how the hit was produced: cosine similarity for dense
    search, a BM25 score for lexical search, a reciprocal-rank score after
    fusion, and a relevance score after reranking. Scores from different
    retrieval modes are not comparable, and the API documents this rather than
    pretending to a single normalised scale.
    """

    chunk_id: str
    document_id: str
    title: str
    text: str
    score: float


def reciprocal_rank_fusion(
    rankings: Sequence[Iterable[SearchHit]],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[SearchHit]:
    """Merge several ranked lists into one using Reciprocal Rank Fusion.

    Each hit contributes ``1 / (k + rank)`` for every list it appears in, with
    ``rank`` starting at 1. A passage that both retrievers rank highly wins; a
    passage only one retriever found can still surface if it ranks near the top.

    The reason to fuse on rank rather than on score is that the scores are not
    comparable: cosine similarity is bounded in [-1, 1] while BM25 is unbounded
    and scales with corpus statistics. Normalising them against each other would
    require assumptions about their distributions that do not hold. Ranks carry
    less information but need no such assumptions, which is the trade-off RRF
    makes deliberately.

    Args:
        rankings: Ranked lists, each already ordered best-first.
        k: Smoothing constant. Larger values flatten the weight given to top
            ranks, making the fusion less sensitive to any single retriever's
            ordering. 60 is the value from the original paper.
        limit: Maximum number of hits to return. ``None`` returns all of them.

    Returns:
        Hits ordered by fused score, with ``score`` replaced by that fused
        value. Ties break on ``chunk_id`` so the output is deterministic.

    Raises:
        ValueError: If ``k`` is negative or ``limit`` is not positive.
    """
    if k < 0:
        raise ValueError("k must not be negative")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    fused: dict[str, float] = {}
    seen: dict[str, SearchHit] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            seen.setdefault(hit.chunk_id, hit)

    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        ordered = ordered[:limit]
    return [replace(seen[chunk_id], score=score) for chunk_id, score in ordered]
