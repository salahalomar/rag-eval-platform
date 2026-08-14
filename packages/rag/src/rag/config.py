"""The single typed description of everything configurable about retrieval.

Why one frozen model rather than arguments threaded through the call stack: the
ablation runner works by instantiating variants of this object, and every evaluation
result embeds the exact instance that produced it. A measurement whose configuration
cannot be reconstructed afterwards is not evidence of anything.

Frozen because a mutable config could be altered by one stage after another stage had
already read it, which would make the copy recorded alongside the result a description
of something that never actually ran. Frozen also makes instances hashable, which the
Phase 7 judgement cache relies on.
"""

from typing import Literal

from pydantic import BaseModel


class RetrievalConfig(BaseModel, frozen=True):
    """One point in the retrieval configuration space.

    Nothing in the retrieval path may read an environment variable; if a knob affects
    what gets retrieved, it belongs here so that it is captured in the result record.
    Process-level concerns such as the database URL live in `rag.settings` instead.
    """

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_tokens: int = 512
    chunk_overlap_pct: float = 0.15
    contextual_headers: bool = True  # prepend paper+section title to chunk before embedding
    dense_enabled: bool = True
    dense_top_k: int = 50
    lexical_enabled: bool = True
    lexical_top_k: int = 50
    fusion: Literal["rrf", "dense_only", "lexical_only"] = "rrf"
    rrf_k: int = 60
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    final_top_k: int = 5
    score_floor: float = 0.0  # below this -> refuse
