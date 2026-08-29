from __future__ import annotations

import pytest

from app.core.lexical import LexicalIndex, tokenize


def build(*passages: tuple[str, str]) -> LexicalIndex:
    index = LexicalIndex()
    for chunk_id, text in passages:
        index.add(chunk_id=chunk_id, document_id="doc", title="Title", text=text)
    return index


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Qdrant, the vector DB!") == ["qdrant", "the", "vector", "db"]


def test_tokenize_keeps_digits_and_accents():
    assert tokenize("Póliza 01847") == ["póliza", "01847"]


def test_exact_term_is_retrieved():
    index = build(("a", "the cat sat"), ("b", "the dog barked"))
    hits = index.search("dog", limit=5)
    assert [hit.chunk_id for hit in hits] == ["b"]


def test_rare_terms_outrank_common_ones():
    """A term in every passage should not decide the ranking; a rare one should."""
    index = build(
        ("common1", "invoice processing overview"),
        ("common2", "invoice processing details"),
        ("common3", "invoice processing summary"),
        ("rare", "invoice ES-2024-01847 was refunded"),
    )
    hits = index.search("invoice 01847", limit=4)
    assert hits[0].chunk_id == "rare"


def test_title_is_indexed_but_only_body_is_returned():
    index = LexicalIndex()
    index.add(chunk_id="a", document_id="d", title="Kubernetes", text="orchestration basics")
    hits = index.search("kubernetes", limit=1)
    assert hits[0].text == "orchestration basics"


def test_shorter_passages_win_at_equal_term_frequency():
    index = build(
        ("short", "qdrant"),
        ("long", "qdrant " + " ".join(f"filler{n}" for n in range(200))),
    )
    hits = index.search("qdrant", limit=2)
    assert hits[0].chunk_id == "short"


def test_limit_bounds_the_result_set():
    index = build(*[(str(n), f"shared term {n}") for n in range(10)])
    assert len(index.search("shared", limit=3)) == 3


def test_unknown_terms_return_nothing():
    index = build(("a", "the cat sat"))
    assert index.search("helicopter", limit=5) == []


def test_empty_index_and_empty_query_return_nothing():
    index = LexicalIndex()
    assert index.search("anything", limit=5) == []
    index.add(chunk_id="a", document_id="d", title="T", text="content")
    assert index.search("!!!", limit=5) == []


def test_readding_the_same_chunk_replaces_it():
    index = LexicalIndex()
    index.add(chunk_id="a", document_id="d", title="T", text="first version")
    index.add(chunk_id="a", document_id="d", title="T", text="second version")

    assert len(index) == 1
    assert index.search("first", limit=5) == []
    assert index.search("second", limit=5)[0].text == "second version"


def test_removing_a_document_drops_its_passages():
    index = LexicalIndex()
    index.add(chunk_id="a", document_id="keep", title="T", text="alpha")
    index.add(chunk_id="b", document_id="drop", title="T", text="beta")
    index.add(chunk_id="c", document_id="drop", title="T", text="gamma")

    assert index.remove_document("drop") == 2
    assert len(index) == 1
    assert index.search("beta", limit=5) == []
    assert index.search("alpha", limit=5)


def test_average_length_tracks_content():
    index = LexicalIndex()
    assert index.average_length == 0.0
    index.add(chunk_id="a", document_id="d", title="", text="one two three")
    assert index.average_length == 3.0
    index.clear()
    assert len(index) == 0
    assert index.average_length == 0.0


@pytest.mark.parametrize(
    ("k1", "b"),
    [(-1.0, 0.75), (1.5, -0.1), (1.5, 1.5)],
)
def test_invalid_parameters_are_rejected(k1: float, b: float):
    with pytest.raises(ValueError):
        LexicalIndex(k1=k1, b=b)


def test_non_positive_limit_is_rejected():
    index = build(("a", "content"))
    with pytest.raises(ValueError):
        index.search("content", limit=0)
