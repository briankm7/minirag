"""Lexical retrieval with BM25.

Dense retrieval fails in a specific, predictable way: it matches meaning, so it
is weak exactly where meaning is not the point. Product codes, error
identifiers, surnames, version numbers and acronyms all embed into vectors that
sit near semantically similar text rather than near the literal string. A query
for ``ES-2024-01847`` retrieves passages about reference numbers in general.

BM25 covers that gap by scoring exact term overlap, weighting rare terms more
heavily than common ones. Neither retriever subsumes the other, which is the
reason to run both and fuse the results.

The index is a plain in-process inverted index with no dependencies. That is a
deliberate limit, not an oversight: see the module notes in the README about
when this stops being adequate.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from app.core.retrieval import SearchHit

_TOKEN = re.compile(r"\w+", re.UNICODE)

# BM25 parameters. k1 controls how quickly repeated terms stop adding value;
# b controls how strongly long passages are penalised. These are the standard
# defaults and are exposed on the constructor rather than hard-coded.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens.

    No stemming and no stopword list. Both would help recall on prose and hurt
    precision on the identifier-style queries this retriever exists to serve,
    and neither can be added without a language assumption the rest of the
    pipeline does not make.
    """
    return _TOKEN.findall(text.lower())


class _Entry:
    """One indexed passage: its metadata, term counts and length."""

    __slots__ = ("chunk_id", "document_id", "title", "text", "terms", "length")

    def __init__(self, chunk_id: str, document_id: str, title: str, text: str) -> None:
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.title = title
        self.text = text
        # The title is indexed alongside the body because it is often the only
        # place a document's subject appears verbatim. Only the body is
        # returned, so this affects ranking without changing what is cited.
        tokens = tokenize(f"{title} {text}")
        self.terms = Counter(tokens)
        self.length = len(tokens)


class LexicalIndex:
    """An in-memory BM25 index over chunk texts.

    Mutation and search are synchronous and contain no ``await``, so on a single
    event loop they cannot interleave. Sharing this index across processes or
    threads would need external coordination; it is scoped to one worker.
    """

    def __init__(self, *, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        if k1 < 0:
            raise ValueError("k1 must not be negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        self._k1 = k1
        self._b = b
        self._entries: dict[str, _Entry] = {}
        self._postings: dict[str, set[str]] = {}
        self._total_length = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def average_length(self) -> float:
        if not self._entries:
            return 0.0
        return self._total_length / len(self._entries)

    def add(self, *, chunk_id: str, document_id: str, title: str, text: str) -> None:
        """Index one passage, replacing any earlier entry with the same id."""
        if chunk_id in self._entries:
            self._discard(chunk_id)
        entry = _Entry(chunk_id, document_id, title, text)
        self._entries[chunk_id] = entry
        self._total_length += entry.length
        for term in entry.terms:
            self._postings.setdefault(term, set()).add(chunk_id)

    def add_many(self, hits: Iterable[SearchHit]) -> None:
        """Index a batch of passages, used to warm the index at startup."""
        for hit in hits:
            self.add(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                title=hit.title,
                text=hit.text,
            )

    def remove_document(self, document_id: str) -> int:
        """Drop every passage belonging to ``document_id``.

        Returns:
            How many passages were removed.
        """
        targets = [
            chunk_id
            for chunk_id, entry in self._entries.items()
            if entry.document_id == document_id
        ]
        for chunk_id in targets:
            self._discard(chunk_id)
        return len(targets)

    def clear(self) -> None:
        self._entries.clear()
        self._postings.clear()
        self._total_length = 0

    def _discard(self, chunk_id: str) -> None:
        entry = self._entries.pop(chunk_id)
        self._total_length -= entry.length
        for term in entry.terms:
            postings = self._postings.get(term)
            if postings is None:
                continue
            postings.discard(chunk_id)
            if not postings:
                del self._postings[term]

    def _idf(self, term: str) -> float:
        """Inverse document frequency, smoothed so it never goes negative.

        A term in every passage carries no discriminating power and lands near
        zero; a term in one passage out of thousands dominates the score.
        """
        total = len(self._entries)
        containing = len(self._postings.get(term, ()))
        return math.log(1 + (total - containing + 0.5) / (containing + 0.5))

    def search(self, query: str, *, limit: int) -> list[SearchHit]:
        """Return the best-scoring passages for ``query``.

        Only passages sharing at least one term with the query are scored, so
        cost tracks the length of the postings lists rather than the size of the
        corpus.

        Raises:
            ValueError: If ``limit`` is not positive.
        """
        if limit <= 0:
            raise ValueError("limit must be positive")
        if not self._entries:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        candidates: set[str] = set()
        for term in set(query_terms):
            candidates |= self._postings.get(term, set())
        if not candidates:
            return []

        average = self.average_length or 1.0
        scored: list[tuple[float, str]] = []
        for chunk_id in candidates:
            entry = self._entries[chunk_id]
            score = 0.0
            for term in set(query_terms):
                frequency = entry.terms.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * entry.length / average
                )
                score += self._idf(term) * frequency * (self._k1 + 1) / denominator
            if score > 0:
                scored.append((score, chunk_id))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            SearchHit(
                chunk_id=chunk_id,
                document_id=self._entries[chunk_id].document_id,
                title=self._entries[chunk_id].title,
                text=self._entries[chunk_id].text,
                score=score,
            )
            for score, chunk_id in scored[:limit]
        ]
