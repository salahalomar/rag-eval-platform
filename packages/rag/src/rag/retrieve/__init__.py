"""Retrieval: dense, lexical, fusion and (from Phase 4) reranking.

`retrieve()` is THE retrieval entry point. The API calls it, the eval harness calls it,
and neither calls anything else in this package. That is the whole argument for the
repository: an evaluation that exercises a parallel implementation measures something
adjacent to what ships, and the gap is invisible until somebody asks about it in an
interview. `tests/test_layering.py` enforces the direction of the dependency; this module
being the only door enforces the rest.

Every arm returns `Candidate`, so fusion combines lists without knowing which retriever
produced them, and the eval runner scores every arm through identical code.
"""

import logging

import psycopg

from rag.config import RetrievalConfig
from rag.index.embed import Encoder
from rag.retrieve import dense, lexical
from rag.retrieve.fusion import reciprocal_rank_fusion
from rag.retrieve.rerank import Reranker
from rag.retrieve.rerank import rerank as rerank_candidates
from rag.retrieve.types import Candidate, RerankStats, RetrievalResult
from rag.telemetry import StageTimer

logger = logging.getLogger(__name__)

__all__ = [
    "Candidate",
    "RerankStats",
    "RetrievalResult",
    "dense",
    "lexical",
    "reciprocal_rank_fusion",
    "rerank_candidates",
    "retrieve",
]


def retrieve(
    query: str,
    config: RetrievalConfig,
    conn: psycopg.Connection,
    *,
    timer: StageTimer | None = None,
    encoder: Encoder | None = None,
    reranker: Reranker | None = None,
) -> RetrievalResult:
    """Run the retrieval pipeline described by `config` and return ranked candidates.

    Dispatch is on `config.fusion` alone, so an ablation arm is a different configuration
    rather than a different code path. `dense_enabled` and `lexical_enabled` gate the arms
    independently of the strategy: fusing over a single enabled arm is a monotonic
    transform of that arm's ordering, so it degrades to that arm rather than erroring.

    `encoder` and `reranker` exist so a caller can supply models rather than have this
    function reach for the real ones. That is what lets the test suite assert routing,
    fusion and floor behaviour deterministically and offline; without it every test
    touching this function would download and run two neural networks to check a branch.
    """
    timer = timer or StageTimer()

    dense_hits: list[Candidate] = []
    lexical_hits: list[Candidate] = []

    want_dense = config.dense_enabled and config.fusion in ("rrf", "dense_only")
    want_lexical = config.lexical_enabled and config.fusion in ("rrf", "lexical_only")

    if want_dense:
        dense_hits = dense.search(conn, query, config, timer=timer, encoder=encoder)
    if want_lexical:
        lexical_hits = lexical.search(conn, query, config, timer=timer)

    if config.fusion == "dense_only":
        fused = dense_hits
    elif config.fusion == "lexical_only":
        fused = lexical_hits
    else:
        with timer.stage("fusion_ms"):
            fused = reciprocal_rank_fusion(
                [hits for hits in (dense_hits, lexical_hits) if hits],
                k=config.rrf_k,
            )

    final, stats = rerank_candidates(query, fused, config, reranker=reranker, timer=timer)

    # The floor gates on the reranker's judgement, which is the only score in the system
    # that reflects the query and the passage together. Refusal is a measured behaviour,
    # not an error path: Phase 5 declines to call the LLM at all on this reason, and the
    # golden set scores how often that was the right call.
    #
    # Worth stating rather than discovering: an arm with reranking disabled cannot refuse
    # on score. Cosine similarity and ts_rank_cd are on scales no single floor value can
    # serve, so gating them against the same number would be arbitrary.
    reason: str | None = None
    if stats is not None and final:
        best = final[0].rerank_score
        if best is not None and best < config.score_floor:
            logger.info(
                "refusing: best rerank score %.3f is below floor %.3f", best, config.score_floor
            )
            reason = "below_score_floor"
            final = []

    result = RetrievalResult(
        candidates=tuple(final),
        dense_count=len(dense_hits),
        lexical_count=len(lexical_hits),
        fused_count=len(fused),
        timings_ms=timer.as_dict(),
        reason=reason,
        rerank_stats=stats,
    )
    logger.debug(
        "retrieve(%s): dense=%d lexical=%d fused=%d returned=%d",
        config.fusion,
        result.dense_count,
        result.lexical_count,
        result.fused_count,
        len(result.candidates),
    )
    return result
