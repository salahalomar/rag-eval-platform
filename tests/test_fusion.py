"""Reciprocal Rank Fusion, checked against values computed by hand.

RRF is four lines of arithmetic, which is exactly why it is worth pinning precisely:
an off-by-one in the rank base or a document silently scored as rank 0 changes every
result in the ablation table while still producing a plausible ordering.
"""

import pytest

from rag.retrieve.fusion import reciprocal_rank_fusion, rrf_score
from rag.retrieve.types import Candidate


def candidate(chunk_id: int, rank: int, *, score: float = 0.0) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        score=score,
        rank=rank,
        content=f"content {chunk_id}",
        section_path=f"section {chunk_id}",
        paper_id=f"paper-{chunk_id}",
        page_start=1,
        page_end=1,
        paper_title=f"Paper {chunk_id}",
    )


def ranked(*chunk_ids: int) -> list[Candidate]:
    """A ranked list, 1-based, in the order given."""
    return [candidate(chunk_id, rank) for rank, chunk_id in enumerate(chunk_ids, start=1)]


# --- the arithmetic ---------------------------------------------------------


def test_single_rank_is_one_over_k_plus_rank() -> None:
    # 1 / (60 + 1) = 0.016393442622950820
    assert rrf_score([1], k=60) == pytest.approx(1 / 61)


def test_ranks_sum_across_retrievers() -> None:
    # 1/(60+1) + 1/(60+3) = 0.016393442... + 0.015873015... = 0.032266458...
    assert rrf_score([1, 3], k=60) == pytest.approx(1 / 61 + 1 / 63)
    assert rrf_score([1, 3], k=60) == pytest.approx(0.03226645, abs=1e-8)


def test_a_document_in_no_list_scores_zero() -> None:
    # Absence contributes nothing -- not a penalty, and not a zero-rank term, which would
    # be 1/k and would outrank genuinely retrieved documents further down.
    assert rrf_score([], k=60) == 0.0


def test_k_damps_the_advantage_of_the_top_rank() -> None:
    # The whole point of k: at 60, rank 1 over rank 2 is worth 0.00026, while agreeing
    # across both arms is worth ~0.016. Agreement dominates position.
    rank_gap = rrf_score([1], k=60) - rrf_score([2], k=60)
    agreement = rrf_score([50, 50], k=60) - rrf_score([50], k=60)
    assert rank_gap == pytest.approx(1 / 61 - 1 / 62, abs=1e-9)
    assert agreement > rank_gap * 10


def test_smaller_k_sharpens_the_top_rank() -> None:
    assert rrf_score([1], k=1) - rrf_score([2], k=1) > rrf_score([1], k=60) - rrf_score([2], k=60)


# --- fusion over lists ------------------------------------------------------


def test_documents_in_both_lists_outrank_documents_in_one() -> None:
    # 7 is rank 2 in both:      1/62 + 1/62 = 0.03225806
    # 1 is rank 1 in dense:     1/61       = 0.01639344
    dense = ranked(1, 7, 3)
    lexical = ranked(9, 7, 4)
    fused = reciprocal_rank_fusion([dense, lexical], k=60)

    assert fused[0].chunk_id == 7
    assert fused[0].score == pytest.approx(2 / 62)
    assert fused[0].rank == 1


def test_hand_computed_ordering_of_a_full_fusion() -> None:
    dense = ranked(1, 2, 3)
    lexical = ranked(3, 1, 4)
    fused = reciprocal_rank_fusion([dense, lexical], k=60)

    # chunk 1: dense rank 1, lexical rank 2 -> 1/61 + 1/62 = 0.03252...
    # chunk 3: dense rank 3, lexical rank 1 -> 1/63 + 1/61 = 0.03226...
    # chunk 2: dense rank 2 only            -> 1/62        = 0.01613...
    # chunk 4: lexical rank 3 only          -> 1/63        = 0.01587...
    assert [c.chunk_id for c in fused] == [1, 3, 2, 4]
    assert [c.score for c in fused] == pytest.approx(
        [1 / 61 + 1 / 62, 1 / 63 + 1 / 61, 1 / 62, 1 / 63]
    )
    assert [c.rank for c in fused] == [1, 2, 3, 4]


def test_a_document_in_one_list_only_still_appears() -> None:
    fused = reciprocal_rank_fusion([ranked(1), ranked(2)], k=60)
    assert sorted(c.chunk_id for c in fused) == [1, 2]
    assert all(c.score == pytest.approx(1 / 61) for c in fused)


def test_exact_ties_break_deterministically_on_chunk_id() -> None:
    # Both documents sit at rank 1 of one list each, so their scores are identical. An
    # unordered tie makes two runs of the same evaluation report different metrics.
    fused = reciprocal_rank_fusion([ranked(9), ranked(4)], k=60)
    assert [c.chunk_id for c in fused] == [4, 9]
    assert fused[0].score == pytest.approx(fused[1].score)

    reversed_inputs = reciprocal_rank_fusion([ranked(4), ranked(9)], k=60)
    assert [c.chunk_id for c in reversed_inputs] == [4, 9]


def test_both_lists_empty_yields_nothing() -> None:
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_no_lists_at_all_yields_nothing() -> None:
    assert reciprocal_rank_fusion([], k=60) == []


def test_one_empty_list_degrades_to_the_other() -> None:
    fused = reciprocal_rank_fusion([ranked(5, 6, 7), []], k=60)
    assert [c.chunk_id for c in fused] == [5, 6, 7]


def test_fusing_a_single_list_preserves_its_order() -> None:
    # Monotonic in rank, so RRF over one arm is that arm. This is what lets `fusion=rrf`
    # with one arm disabled degrade gracefully instead of erroring.
    fused = reciprocal_rank_fusion([ranked(11, 22, 33, 44)], k=60)
    assert [c.chunk_id for c in fused] == [11, 22, 33, 44]


def test_limit_truncates_after_ranking_not_before() -> None:
    dense = ranked(1, 2, 3, 4)
    lexical = ranked(4, 3, 2, 1)
    full = reciprocal_rank_fusion([dense, lexical], k=60)
    limited = reciprocal_rank_fusion([dense, lexical], k=60, limit=2)
    assert [c.chunk_id for c in limited] == [c.chunk_id for c in full][:2]


# --- structural properties --------------------------------------------------


def test_output_never_exceeds_the_union_of_inputs() -> None:
    lists = [ranked(1, 2, 3), ranked(3, 4), ranked(5, 1, 6, 2)]
    union = {c.chunk_id for ranking in lists for c in ranking}
    fused = reciprocal_rank_fusion(lists, k=60)
    assert len(fused) == len(union)
    assert {c.chunk_id for c in fused} == union


def test_scores_are_non_increasing_and_ranks_are_contiguous() -> None:
    fused = reciprocal_rank_fusion([ranked(1, 2, 3, 4), ranked(4, 1, 5)], k=60)
    assert [c.score for c in fused] == sorted((c.score for c in fused), reverse=True)
    assert [c.rank for c in fused] == list(range(1, len(fused) + 1))


def test_each_chunk_appears_exactly_once() -> None:
    fused = reciprocal_rank_fusion([ranked(1, 2, 3), ranked(3, 2, 1), ranked(2, 1, 3)], k=60)
    assert len(fused) == len({c.chunk_id for c in fused}) == 3


def test_metadata_comes_from_the_arm_that_ranked_it_highest() -> None:
    # The candidates carry differing content so the winner is identifiable. Taking the
    # better-ranked arm's copy keeps the displayed text consistent with why it was chosen.
    weak = [
        Candidate(
            chunk_id=5,
            score=0.1,
            rank=9,
            content="from weak arm",
            section_path="s",
            paper_id="p",
            page_start=1,
            page_end=1,
        )
    ]
    strong = [
        Candidate(
            chunk_id=5,
            score=0.9,
            rank=1,
            content="from strong arm",
            section_path="s",
            paper_id="p",
            page_start=1,
            page_end=1,
        )
    ]
    assert reciprocal_rank_fusion([weak, strong], k=60)[0].content == "from strong arm"
    assert reciprocal_rank_fusion([strong, weak], k=60)[0].content == "from strong arm"


def test_fused_score_replaces_the_arm_score() -> None:
    # A caller reading `.score` must see the number that produced the ordering in front
    # of it, not a leftover cosine similarity from one of the arms.
    fused = reciprocal_rank_fusion([ranked(1)], k=60)
    assert fused[0].score == pytest.approx(1 / 61)


@pytest.mark.parametrize("k", [1, 10, 60, 120, 1000])
def test_ordering_is_stable_across_k_for_unanimous_lists(k: int) -> None:
    # When both arms agree exactly, no choice of k can reorder the result -- a useful
    # sanity property, since k is swept as an ablation arm in Phase 7.
    fused = reciprocal_rank_fusion([ranked(1, 2, 3), ranked(1, 2, 3)], k=k)
    assert [c.chunk_id for c in fused] == [1, 2, 3]
