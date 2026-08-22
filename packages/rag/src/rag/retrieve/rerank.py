"""Cross-encoder reranking.

A bi-encoder embeds the query and the passage independently, so it can never see how
they relate -- the two vectors are computed before either knows the other exists. A cross
-encoder concatenates them into a single input and attends across the pair, which is why
it consistently reorders results a bi-encoder got wrong, and why it cannot be indexed:
every (query, passage) pair needs its own forward pass. That is the whole trade. Fifty
forward passes on CPU per query is the price of the largest single quality jump in the
system, and the ablation table shows both columns rather than only the flattering one.

Two things here are worth stating rather than assuming.

**Scores are raw logits, not probabilities.** bge-reranker-base emits roughly [-11, +11],
where 0 is the neutral point. `score_floor` compares against that directly, so the default
of 0.0 means "refuse when even the best passage is below neutral relevance". Squashing
through a sigmoid would put every score above zero and silently turn that default into
"never refuse".

**The 512-token window truncates.** The model encodes `[CLS] query [SEP] passage [SEP]`,
and chunks in this corpus have a median of 477 tokens. Pairs therefore overflow, and the
passage tail is dropped. That is inherent to the model, not a bug to fix here -- but it is
measured (`truncated_pairs`) rather than left invisible, because a reranker silently
judging the first two-thirds of a chunk is exactly the kind of thing that quietly caps
recall.
"""

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from rag.config import RetrievalConfig
from rag.retrieve.types import Candidate, RerankStats
from rag.telemetry import StageTimer

logger = logging.getLogger(__name__)

# The cross-encoder's position limit, including [CLS] and the two [SEP]s.
MODEL_MAX_TOKENS = 512
DEFAULT_BATCH_SIZE = 16
TORCH_SEED = 0


class Reranker(Protocol):
    """Scores (query, passage) pairs by relevance."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance logits for each passage against `query`, in the order given."""
        ...


class CrossEncoderReranker:
    """bge-reranker-base via sentence-transformers, pinned to CPU and to eval mode."""

    def __init__(self, model_name: str, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        """Load `model_name` and put it in a deterministic, inference-only state."""
        import torch
        from sentence_transformers import CrossEncoder

        torch.manual_seed(TORCH_SEED)
        torch.set_grad_enabled(False)

        self._model_name = model_name
        self._batch_size = batch_size
        # max_length is set explicitly rather than left to the library default, which
        # varies by version. An implicit shorter window would truncate more of every
        # passage than intended and nothing would say so.
        self._model = CrossEncoder(model_name, max_length=MODEL_MAX_TOKENS, device="cpu")
        self._model.model.eval()

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance logits for each passage against `query`."""
        if not passages:
            return []
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(value) for value in scores]

    def count_truncated(self, query: str, passages: Sequence[str]) -> int:
        """How many pairs exceed the model's window and lose their tail."""
        tokenizer = self._model.tokenizer
        return sum(
            1
            for passage in passages
            if len(tokenizer(query, passage)["input_ids"]) > MODEL_MAX_TOKENS
        )

    def __repr__(self) -> str:
        """Identify the model behind the scores."""
        return f"CrossEncoderReranker({self._model_name!r})"


@lru_cache(maxsize=2)
def reranker_for(model: str) -> CrossEncoderReranker:
    """Cached reranker, because loading the model dominates a single query."""
    return CrossEncoderReranker(model)


def rerank(
    query: str,
    candidates: Sequence[Candidate],
    config: RetrievalConfig,
    *,
    reranker: Reranker | None = None,
    timer: StageTimer | None = None,
) -> tuple[list[Candidate], RerankStats | None]:
    """Reorder `candidates` by cross-encoder relevance and keep the best `final_top_k`.

    A pass-through when `config.rerank_enabled` is false -- the candidates come back
    untouched and stats are None, so a caller can always tell "reranking ran and moved
    nothing" apart from "reranking never ran".

    Only the top `rerank_top_n` of the fused list is scored. The whole fused list can run
    to a hundred candidates, and a hundred CPU forward passes per query costs seconds for
    ordering that is thrown away below rank five anyway.
    """
    timer = timer or StageTimer()

    if not config.rerank_enabled or not candidates:
        return list(candidates[: config.final_top_k]), None

    shortlist = list(candidates[: config.rerank_top_n])
    reranker = reranker or reranker_for(config.rerank_model)

    with timer.stage("rerank_ms"):
        scores = reranker.score(query, [candidate.content for candidate in shortlist])

    scored = list(zip(shortlist, scores, strict=True))
    # Ties break on chunk_id, as everywhere else: the cross-encoder returns float32
    # logits, and exact ties across near-identical passages are not rare.
    scored.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))

    reranked = [
        Candidate(
            chunk_id=candidate.chunk_id,
            score=candidate.score,
            rank=position,
            content=candidate.content,
            section_path=candidate.section_path,
            paper_id=candidate.paper_id,
            page_start=candidate.page_start,
            page_end=candidate.page_end,
            paper_title=candidate.paper_title,
            char_start=candidate.char_start,
            char_end=candidate.char_end,
            rerank_score=score,
        )
        for position, (candidate, score) in enumerate(scored, start=1)
    ]

    final = reranked[: config.final_top_k]
    stats = RerankStats(
        scored=len(shortlist),
        truncated_pairs=_count_truncated(reranker, query, shortlist),
        mean_rank_movement=_mean_rank_movement(shortlist, final),
        max_rank_movement=_max_rank_movement(shortlist, final),
    )
    logger.debug(
        "reranked %d candidates, mean movement %.1f, %d truncated",
        stats.scored,
        stats.mean_rank_movement,
        stats.truncated_pairs,
    )
    return final, stats


def _rank_before(shortlist: Sequence[Candidate]) -> dict[int, int]:
    return {candidate.chunk_id: candidate.rank for candidate in shortlist}


def _movements(shortlist: Sequence[Candidate], final: Sequence[Candidate]) -> list[int]:
    """How far each surviving candidate moved between fusion order and rerank order.

    Measured over the candidates that actually survive into the final list, because that
    is the question worth answering: not "did the ordering churn", but "did reranking
    change what the user is shown".
    """
    before = _rank_before(shortlist)
    return [abs(before[candidate.chunk_id] - candidate.rank) for candidate in final]


def _mean_rank_movement(shortlist: Sequence[Candidate], final: Sequence[Candidate]) -> float:
    movements = _movements(shortlist, final)
    return sum(movements) / len(movements) if movements else 0.0


def _max_rank_movement(shortlist: Sequence[Candidate], final: Sequence[Candidate]) -> int:
    movements = _movements(shortlist, final)
    return max(movements) if movements else 0


def _count_truncated(reranker: Reranker, query: str, shortlist: Sequence[Candidate]) -> int:
    """Truncation count, when the reranker can report it.

    A fake reranker in a test has no tokenizer and no window, so this degrades to zero
    rather than forcing every test double to implement a method it has no opinion about.
    """
    counter = getattr(reranker, "count_truncated", None)
    if not callable(counter):
        return 0
    count: int = counter(query, [candidate.content for candidate in shortlist])
    return count
