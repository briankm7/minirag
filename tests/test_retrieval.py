from __future__ import annotations

import pytest

from app.core.retrieval import SearchHit, reciprocal_rank_fusion


def hit(chunk_id: str, score: float = 0.0) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id="doc",
        title="Title",
        text=f"text {chunk_id}",
        score=score,
    )


def test_agreement_between_rankings_wins():
    """A passage both retrievers rank well beats one only a single retriever tops.

    Note the strength of the effect: appearing second in both lists beats
    appearing first in one. That is the property fusion is bought for.
    """
    dense = [hit("a"), hit("b"), hit("c")]
    lexical = [hit("d"), hit("b"), hit("e")]

    fused = reciprocal_rank_fusion([dense, lexical])
    assert fused[0].chunk_id == "b"


def test_all_unique_hits_survive_fusion():
    fused = reciprocal_rank_fusion([[hit("a"), hit("b")], [hit("c")]])
    assert {h.chunk_id for h in fused} == {"a", "b", "c"}


def test_a_hit_found_by_only_one_retriever_can_still_rank_first():
    """Fusion must not require agreement, or it would just be an intersection."""
    dense = [hit("x"), hit("y"), hit("z")]
    lexical = [hit("only")]

    fused = reciprocal_rank_fusion([dense, lexical])
    assert fused[0].chunk_id in {"x", "only"}
    assert "only" in {h.chunk_id for h in fused}


def test_score_is_replaced_by_the_fused_value():
    """Incoming scores are on incomparable scales and must not leak through."""
    fused = reciprocal_rank_fusion([[hit("a", score=0.99)]], k=60)
    assert fused[0].score == pytest.approx(1 / 61)


def test_single_ranking_preserves_its_order():
    fused = reciprocal_rank_fusion([[hit("a"), hit("b"), hit("c")]])
    assert [h.chunk_id for h in fused] == ["a", "b", "c"]


def test_limit_truncates_the_fused_ranking():
    fused = reciprocal_rank_fusion([[hit("a"), hit("b"), hit("c")]], limit=2)
    assert len(fused) == 2


def test_ties_break_deterministically():
    first = reciprocal_rank_fusion([[hit("b")], [hit("a")]])
    second = reciprocal_rank_fusion([[hit("b")], [hit("a")]])
    assert [h.chunk_id for h in first] == [h.chunk_id for h in second] == ["a", "b"]


def test_larger_k_flattens_the_weight_of_top_ranks():
    dense = [hit("a"), hit("b")]
    lexical = [hit("a"), hit("b")]

    sharp = reciprocal_rank_fusion([dense, lexical], k=1)
    flat = reciprocal_rank_fusion([dense, lexical], k=1000)
    gap_sharp = sharp[0].score - sharp[1].score
    gap_flat = flat[0].score - flat[1].score
    assert gap_flat < gap_sharp


def test_empty_input_yields_nothing():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


@pytest.mark.parametrize(("k", "limit"), [(-1, None), (60, 0), (60, -3)])
def test_invalid_parameters_are_rejected(k: int, limit: int | None):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[hit("a")]], k=k, limit=limit)
