r"""Reciprocal Rank Fusion.

    RRF(d) = sum over retrievers i of  1 / (k + rank_i(d))

Why rank fusion rather than score fusion: the two arms produce numbers that are not
comparable. Cosine similarity is a bounded geometric quantity; `ts_rank_cd` is an
unbounded coverage-density statistic that has been squashed into [0,1) by a normalisation
flag. Any attempt to combine them by value -- min-max scaling, z-scores, a weighted sum --
requires inventing an exchange rate between two things that do not have one, and that
exchange rate then silently becomes a tuned hyperparameter nobody measured. Ranks discard
the magnitudes entirely, which is the point.

The `k` constant damps the influence of the top ranks. At k=60 the difference between
rank 1 and rank 2 is 1/61 - 1/62 = 0.00026, while the difference between rank 1 and rank
50 is 0.0073 -- so agreement between arms matters more than either arm's exact ordering. A
document missing from a retriever's list contributes nothing from that retriever, rather
than contributing a penalty, which is what lets lists of different lengths be fused
without normalisation.

Pure functions over rank lists. No database access, no configuration reading, no I/O --
so the arithmetic can be unit-tested against values computed by hand, which is the only
way anyone can check it is right.
"""

import logging
from collections.abc import Sequence

from rag.retrieve.types import Candidate

logger = logging.getLogger(__name__)


def rrf_score(ranks: Sequence[int], k: int) -> float:
    """Sum of reciprocal ranks for one document across the retrievers that returned it.

    `ranks` are 1-based. Retrievers that did not return the document are simply absent
    from the sequence -- they contribute nothing rather than contributing a zero-rank or
    a penalty.
    """
    return sum(1.0 / (k + rank) for rank in ranks)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Candidate]],
    *,
    k: int,
    limit: int | None = None,
) -> list[Candidate]:
    """Fuse ranked candidate lists into one, ordered by descending RRF score.

    Ties break on `chunk_id`. RRF produces exact ties routinely -- any two documents
    appearing at the same rank in the same number of lists score identically -- and
    without a deterministic tiebreak two runs of the same evaluation would order them
    differently and report different metrics.

    Each surviving candidate keeps the metadata from the arm that ranked it highest, and
    is renumbered to its position in the fused list; its `score` becomes the RRF score, so
    what a caller reads is always the score that produced the ordering it sees.
    """
    ranks_by_chunk: dict[int, list[int]] = {}
    best_by_chunk: dict[int, Candidate] = {}

    for ranking in rankings:
        for candidate in ranking:
            ranks_by_chunk.setdefault(candidate.chunk_id, []).append(candidate.rank)
            incumbent = best_by_chunk.get(candidate.chunk_id)
            if incumbent is None or candidate.rank < incumbent.rank:
                best_by_chunk[candidate.chunk_id] = candidate

    scored = [(rrf_score(ranks, k), chunk_id) for chunk_id, ranks in sorted(ranks_by_chunk.items())]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))

    fused = [
        Candidate(
            chunk_id=chunk_id,
            score=score,
            rank=position,
            content=best_by_chunk[chunk_id].content,
            section_path=best_by_chunk[chunk_id].section_path,
            paper_id=best_by_chunk[chunk_id].paper_id,
            page_start=best_by_chunk[chunk_id].page_start,
            page_end=best_by_chunk[chunk_id].page_end,
            paper_title=best_by_chunk[chunk_id].paper_title,
            char_start=best_by_chunk[chunk_id].char_start,
            char_end=best_by_chunk[chunk_id].char_end,
        )
        for position, (score, chunk_id) in enumerate(scored, start=1)
    ]

    if limit is not None:
        fused = fused[:limit]
    logger.debug("fused %d lists into %d candidates (k=%d)", len(rankings), len(fused), k)
    return fused
