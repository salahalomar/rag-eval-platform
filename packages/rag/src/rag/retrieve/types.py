"""The shape every retrieval arm returns.

One type across dense, lexical, fusion and rerank, so that fusion can combine arms
without knowing which produced what, and so the eval runner scores every arm through
identical code. An arm returning its own bespoke shape is how a fusion implementation
ends up quietly favouring whichever arm it was written against first.
"""

from dataclasses import dataclass, replace


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
