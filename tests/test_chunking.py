from __future__ import annotations

import pytest

from app.core.chunking import chunk_text, normalize


def test_normalize_collapses_whitespace_but_keeps_paragraphs():
    assert normalize("a   b\n\n  c \t d ") == "a b\n\nc d"


def test_empty_input_yields_no_chunks():
    assert chunk_text("   \n\n  ", chunk_size=10, overlap=2) == []


def test_short_text_fits_in_one_chunk():
    chunks = chunk_text("one two three", chunk_size=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0].text == "one two three"
    assert chunks[0].index == 0


def test_long_text_is_split_with_overlap():
    text = " ".join(str(n) for n in range(100))
    chunks = chunk_text(text, chunk_size=40, overlap=10)

    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Every chunk respects the size limit.
    assert all(len(c.text.split()) <= 40 for c in chunks)
    # Consecutive chunks genuinely overlap.
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-10:] == second_words[:10]


def test_no_content_is_lost():
    text = " ".join(str(n) for n in range(95))
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    seen = {word for chunk in chunks for word in chunk.text.split()}
    assert seen == set(text.split())


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 20)],
)
def test_invalid_parameters_are_rejected(chunk_size: int, overlap: int):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=chunk_size, overlap=overlap)
