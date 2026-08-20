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
from rag.retrieve import dense, lexical
from rag.retrieve.fusion import reciprocal_rank_fusion
from rag.retrieve.types import Candidate, RetrievalResult
from rag.telemetry import StageTimer

logger = logging.getLogger(__name__)

__all__ = [
    "Candidate",
    "RetrievalResult",
    "dense",
    "lexical",
    "reciprocal_rank_fusion",
    "retrieve",
]


def retrieve(
    query: str,
    config: RetrievalConfig,
    conn: psycopg.Connection,
    *,
    timer: StageTimer | None = None,
) -> RetrievalResult:
    """Run the retrieval pipeline described by `config` and return ranked candidates.

    Dispatch is on `config.fusion` alone, so an ablation arm is a different configuration
    rather than a different code path. `dense_enabled` and `lexical_enabled` gate the arms
    independently of the strategy: fusing over a single enabled arm is a monotonic
    transform of that arm's ordering, so it degrades to that arm rather than erroring.

    Reranking arrives in Phase 4 and slots in after fusion, consuming the same list.
    """
    timer = timer or StageTimer()

    dense_hits: list[Candidate] = []
    lexical_hits: list[Candidate] = []

    want_dense = config.dense_enabled and config.fusion in ("rrf", "dense_only")
    want_lexical = config.lexical_enabled and config.fusion in ("rrf", "lexical_only")

    if want_dense:
        dense_hits = dense.search(conn, query, config, timer=timer)
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

    result = RetrievalResult(
        candidates=tuple(fused[: config.final_top_k]),
        dense_count=len(dense_hits),
        lexical_count=len(lexical_hits),
        fused_count=len(fused),
        timings_ms=timer.as_dict(),
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
