"""The shape every retrieval arm returns.

One type across dense, lexical, fusion and rerank, so that fusion can combine arms
without knowing which produced what, and so the eval runner scores every arm through
identical code. An arm returning its own bespoke shape is how a fusion implementation
ends up quietly favouring whichever arm it was written against first.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field, replace


@dataclass(frozen=True, slots=True)
class Candidate:
    """One retrieved chunk, with everything needed to rank, cite and display it."""

    chunk_id: int
    score: float
    rank: int  # 1-based position within the arm that produced this candidate
    content: str
    section_path: str
    paper_id: str
    page_start: int
    page_end: int

    # Not in the original specification, and both are here to avoid a second query later:
    # Phase 5 puts the paper title in the prompt's context blocks, and Phase 8 highlights
    # the character span in the source. Fetching them now costs one join already being
    # performed.
    paper_title: str = ""
    char_start: int = 0
    char_end: int = 0

    # Set by the reranker in Phase 4; None means this candidate never passed through one.
    rerank_score: float | None = None

    def at_rank(self, rank: int) -> "Candidate":
        """Copy of this candidate repositioned at `rank`, for use after reordering."""
        return replace(self, rank=rank)


@dataclass(frozen=True, slots=True)
class RerankStats:
    """What reranking did, so its effect is reported rather than assumed.

    Lives here beside `RetrievalResult` rather than in `rerank`, because the result
    carries it and the reranker imports the candidate types -- putting it the other way
    round makes the two modules import each other.
    """

    scored: int
    truncated_pairs: int
    mean_rank_movement: float
    max_rank_movement: int

    @property
    def truncation_rate(self) -> float:
        """Share of pairs longer than the model's window, whose tail was dropped."""
        return self.truncated_pairs / self.scored if self.scored else 0.0


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """What one call to `retrieve()` produced, and what it cost.

    Carries the per-arm counts alongside the candidates because they are the first thing
    worth seeing when a result looks wrong: a fused list of five that came from fifty
    dense hits and zero lexical hits is a lexical arm that failed, not a fusion that
    chose. Reporting only the final list hides that completely.

    Timings ride along for the same reason cost and latency are treated as features:
    every retrieval path records its per-stage milliseconds, and `query_logs` stores them
    verbatim.
    """

    candidates: tuple[Candidate, ...]
    dense_count: int = 0
    lexical_count: int = 0
    fused_count: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)

    # Why the candidate list is empty, when it is. Phase 5 refuses without calling the
    # LLM on `below_score_floor`, and that distinction has to survive out of retrieval:
    # "the corpus has nothing relevant" and "the query matched nothing at all" are
    # different failures, and only one of them is the system working correctly.
    reason: str | None = None

    # Present only when a reranker actually ran. None means it never did, which is a
    # different statement from "it ran and moved nothing".
    rerank_stats: RerankStats | None = None

    @property
    def refused(self) -> bool:
        """Whether retrieval declined to return anything it had, rather than finding none."""
        return self.reason == "below_score_floor"

    def __len__(self) -> int:
        """Number of candidates returned, so callers can treat this as a sequence."""
        return len(self.candidates)

    def __iter__(self) -> Iterator[Candidate]:
        """Iterate the candidates directly, which is what most callers want."""
        return iter(self.candidates)

    @property
    def chunk_ids(self) -> list[int]:
        """Ranked chunk ids, the form the evaluation metrics consume."""
        return [candidate.chunk_id for candidate in self.candidates]
