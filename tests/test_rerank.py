"""Cross-encoder reranking: reordering, the score floor, and rank-movement arithmetic.

The reranker is faked. These tests assert what the *stage* does with scores -- reorder,
truncate to final_top_k, gate on the floor, measure movement -- not what the model
believes. Tests that need the real cross-encoder carry the `model` marker.
"""

import pytest

from conftest import ConstantReranker, WordOverlapReranker
from rag.config import RetrievalConfig
from rag.retrieve.rerank import MODEL_MAX_TOKENS, rerank
from rag.retrieve.types import Candidate, RerankStats

CONFIG = RetrievalConfig(rerank_enabled=True, rerank_top_n=50, final_top_k=3)


def candidate(chunk_id: int, rank: int, content: str) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        score=1.0 / rank,
        rank=rank,
        content=content,
        section_path="s",
        paper_id="p",
        page_start=1,
        page_end=1,
    )


def fused(*contents: str) -> list[Candidate]:
    """A fused list in the given order, ranked 1..n with chunk ids 101, 102, ..."""
    return [candidate(100 + index, index, text) for index, text in enumerate(contents, start=1)]


# --- pass-through -----------------------------------------------------------


def test_disabled_reranking_is_a_pass_through() -> None:
    candidates = fused("alpha", "beta", "gamma", "delta")
    result, stats = rerank("alpha", candidates, CONFIG.model_copy(update={"rerank_enabled": False}))
    assert [c.chunk_id for c in result] == [101, 102, 103]
    assert stats is None


def test_stats_are_none_when_disabled_and_present_when_run() -> None:
    # The distinction matters: "the reranker ran and moved nothing" is a finding,
    # "the reranker never ran" is a configuration.
    candidates = fused("alpha", "beta")
    _, off = rerank("alpha", candidates, CONFIG.model_copy(update={"rerank_enabled": False}))
    _, on = rerank("alpha", candidates, CONFIG, reranker=WordOverlapReranker())
    assert off is None
    assert isinstance(on, RerankStats)


def test_an_empty_candidate_list_is_handled() -> None:
    result, stats = rerank("anything", [], CONFIG, reranker=WordOverlapReranker())
    assert result == []
    assert stats is None


def test_disabled_reranking_still_honours_final_top_k() -> None:
    candidates = fused("a", "b", "c", "d", "e")
    result, _ = rerank("a", candidates, CONFIG.model_copy(update={"rerank_enabled": False}))
    assert len(result) == 3


# --- reordering -------------------------------------------------------------


def test_reranking_reorders_by_score_not_by_fusion_rank() -> None:
    # Fusion put "nothing relevant" first; the reranker sees the query terms in the
    # third candidate and promotes it. This is the whole point of the stage.
    candidates = fused(
        "nothing relevant here at all",
        "one warmup mention",
        "warmup warmup warmup learning rate",
    )
    result, _ = rerank("warmup learning rate", candidates, CONFIG, reranker=WordOverlapReranker())
    assert [c.chunk_id for c in result] == [103, 102, 101]


def test_rerank_score_is_attached_and_fusion_score_is_preserved() -> None:
    # Both are kept on purpose: Phase 8's config panel shows what fusion thought and what
    # the reranker thought, and the difference is the demo.
    candidates = fused("alpha beta", "gamma")
    result, _ = rerank("alpha beta", candidates, CONFIG, reranker=WordOverlapReranker())
    top = result[0]
    assert top.rerank_score == pytest.approx(2.0)
    assert top.score == pytest.approx(1.0)  # 1/rank from the fused list


def test_ranks_are_renumbered_contiguously_after_reordering() -> None:
    candidates = fused("x", "query", "query query", "y")
    result, _ = rerank("query", candidates, CONFIG, reranker=WordOverlapReranker())
    assert [c.rank for c in result] == [1, 2, 3]


def test_final_top_k_truncates_after_reranking_not_before() -> None:
    # Truncating first would discard the candidate the reranker was going to promote,
    # which is the failure mode that makes a reranker look useless.
    candidates = fused("no", "no", "no", "no", "query query query")
    config = CONFIG.model_copy(update={"final_top_k": 1})
    result, _ = rerank("query", candidates, config, reranker=WordOverlapReranker())
    assert [c.chunk_id for c in result] == [105]


def test_ties_break_deterministically_on_chunk_id() -> None:
    candidates = fused("query", "query", "query")
    result, _ = rerank("query", candidates, CONFIG, reranker=WordOverlapReranker())
    assert [c.chunk_id for c in result] == [101, 102, 103]

    reversed_order = list(reversed(candidates))
    again, _ = rerank("query", reversed_order, CONFIG, reranker=WordOverlapReranker())
    assert [c.chunk_id for c in again] == [101, 102, 103]


def test_only_rerank_top_n_candidates_are_scored() -> None:
    # A hundred CPU forward passes per query costs seconds, for ordering discarded below
    # rank five. The bound is what makes the stage affordable.
    spy = WordOverlapReranker()
    candidates = fused(*[f"text {i}" for i in range(40)])
    rerank("text", candidates, CONFIG.model_copy(update={"rerank_top_n": 10}), reranker=spy)
    assert spy.calls == [("text", 10)]


def test_a_candidate_outside_rerank_top_n_cannot_be_promoted() -> None:
    spy = WordOverlapReranker()
    candidates = fused("no", "no", "query query query")
    config = CONFIG.model_copy(update={"rerank_top_n": 2})
    result, _ = rerank("query", candidates, config, reranker=spy)
    assert 103 not in [c.chunk_id for c in result]


# --- the score floor --------------------------------------------------------


def test_a_top_score_below_the_floor_refuses() -> None:
    candidates = fused("a", "b", "c")
    config = CONFIG.model_copy(update={"score_floor": 1.0})
    result, stats = rerank("zzz", candidates, config, reranker=ConstantReranker(-2.0))
    # rerank() itself returns the ranked list; the floor is applied by retrieve(), which
    # is where the decision has to be visible to the caller.
    assert result
    assert stats is not None
    assert all(c.rerank_score == pytest.approx(-2.0) for c in result)


def test_scores_can_be_negative() -> None:
    # bge-reranker-base emits logits, not probabilities. A floor of 0.0 is meaningful
    # precisely because scores straddle zero; a sigmoid would destroy that.
    candidates = fused("a", "b")
    result, _ = rerank("q", candidates, CONFIG, reranker=ConstantReranker(-5.5))
    assert all(c.rerank_score is not None and c.rerank_score < 0 for c in result)


# --- rank movement ----------------------------------------------------------


def test_rank_movement_is_zero_when_the_reranker_agrees_with_fusion() -> None:
    candidates = fused("query query query", "query query", "query")
    _, stats = rerank("query", candidates, CONFIG, reranker=WordOverlapReranker())
    assert stats is not None
    assert stats.mean_rank_movement == 0.0
    assert stats.max_rank_movement == 0


def test_rank_movement_measures_a_full_reversal() -> None:
    # Fusion ranks 1,2,3; the reranker reverses them to 3,2,1. Movements are |1-3|=2,
    # |2-2|=0, |3-1|=2 -> mean 4/3.
    candidates = fused("query", "query query", "query query query")
    _, stats = rerank("query", candidates, CONFIG, reranker=WordOverlapReranker())
    assert stats is not None
    assert stats.max_rank_movement == 2
    assert stats.mean_rank_movement == pytest.approx(4 / 3)


def test_stats_report_how_many_were_scored() -> None:
    candidates = fused(*[f"text {i}" for i in range(20)])
    _, stats = rerank("text", candidates, CONFIG, reranker=WordOverlapReranker())
    assert stats is not None
    assert stats.scored == 20


def test_truncation_rate_is_zero_when_the_reranker_cannot_report_it() -> None:
    # A fake has no tokenizer and no window; degrading to zero beats forcing every test
    # double to implement a method it has no opinion about.
    candidates = fused("a", "b")
    _, stats = rerank("q", candidates, CONFIG, reranker=WordOverlapReranker())
    assert stats is not None
    assert stats.truncated_pairs == 0
    assert stats.truncation_rate == 0.0


def test_truncation_rate_is_a_share_of_pairs_scored() -> None:
    assert RerankStats(
        scored=50, truncated_pairs=20, mean_rank_movement=0, max_rank_movement=0
    ).truncation_rate == pytest.approx(0.4)
    assert (
        RerankStats(
            scored=0, truncated_pairs=0, mean_rank_movement=0, max_rank_movement=0
        ).truncation_rate
        == 0.0
    )


# --- the real model ---------------------------------------------------------


@pytest.mark.model
def test_the_real_cross_encoder_prefers_the_relevant_passage() -> None:
    """Requires model weights. The one assertion worth making about the model itself."""
    from rag.retrieve.rerank import reranker_for

    model = reranker_for("BAAI/bge-reranker-base")
    scores = model.score(
        "What is the effect of learning rate warmup on transformer training?",
        [
            "Cats are small domesticated carnivorous mammals kept as pets.",
            "Learning rate warmup gradually increases the step size over the first few "
            "thousand updates, which stabilises transformer training and prevents early "
            "divergence in the attention layers.",
        ],
    )
    assert scores[1] > scores[0]


@pytest.mark.model
def test_the_real_cross_encoder_is_deterministic() -> None:
    """Two runs of `make eval` must agree, and reranking is on that path."""
    from rag.retrieve.rerank import reranker_for

    model = reranker_for("BAAI/bge-reranker-base")
    passages = ["a passage about retrieval", "a passage about something else"]
    assert model.score("retrieval", passages) == model.score("retrieval", passages)


@pytest.mark.model
def test_the_real_cross_encoder_reports_truncated_pairs() -> None:
    """A long passage overflows the 512-token window and must be counted, not hidden."""
    from rag.retrieve.rerank import reranker_for

    model = reranker_for("BAAI/bge-reranker-base")
    long_passage = "token " * (MODEL_MAX_TOKENS + 200)
    assert model.count_truncated("query", [long_passage]) == 1
    assert model.count_truncated("query", ["short passage"]) == 0
