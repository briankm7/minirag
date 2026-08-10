"""Split raw text into overlapping chunks suitable for embedding.

Chunking is deliberately kept dependency-free and deterministic: it is the
component most often responsible for poor retrieval quality, so it must be
easy to reason about and easy to test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a source document."""

    text: str
    index: int


def normalize(text: str) -> str:
    """Collapse runs of whitespace while preserving paragraph boundaries."""
    paragraphs = (_WHITESPACE.sub(" ", p).strip() for p in _PARAGRAPH_BREAK.split(text))
    return "\n\n".join(p for p in paragraphs if p)


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[Chunk]:
    """Split ``text`` into word windows of ``chunk_size`` with ``overlap``.

    Args:
        text: Raw document text.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words repeated between consecutive chunks, which
            prevents a sentence spanning a boundary from being lost.

    Returns:
        Chunks in document order. Empty input yields an empty list.

    Raises:
        ValueError: If ``chunk_size`` is not positive, or ``overlap`` is
            negative or greater than or equal to ``chunk_size`` (which would
            make the window advance by zero words and loop forever).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = normalize(text).split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(Chunk(text=" ".join(window), index=len(chunks)))
        if start + chunk_size >= len(words):
            break
    return chunks
