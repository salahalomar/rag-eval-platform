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

import hashlib
import json
from typing import Any, ClassVar, Literal

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
    drop_references: bool = True  # references sections answer no question worth asking
    drop_figure_only_pages: bool = True  # pages whose extractable text is a caption or less
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

    # Fields that change what a chunk *is*, as opposed to how chunks are searched.
    # Listed once, here, because two things depend on getting the set exactly right:
    # the `chunks.chunk_config` record, and the identity under which a chunking is
    # stored and later retrieved. Omitting a field that does affect chunking would let
    # two different chunkings collide under one identity.
    CHUNKING_FIELDS: ClassVar[tuple[str, ...]] = (
        "embedding_model",  # selects the tokenizer, so it moves every chunk boundary
        "chunk_tokens",
        "chunk_overlap_pct",
        "drop_references",
        "drop_figure_only_pages",
        "contextual_headers",  # changes embed_input, not content
    )

    # Bumped whenever the ingestion algorithm changes in a way that moves chunk
    # boundaries: a parser fix, a new heading heuristic, a different sentence splitter.
    #
    # Configuration alone cannot express "the code that produced this". Without this
    # field, a corpus re-ingested after a chunker fix is skipped as unchanged, and the
    # database quietly keeps serving chunks built by the old, broken logic.
    CHUNKING_PIPELINE_VERSION: ClassVar[int] = 1

    def chunking_params(self) -> dict[str, Any]:
        """The inputs that determine chunk boundaries and embed inputs.

        Persisted verbatim into `chunks.chunk_config` so a chunk can always be traced
        back to both the settings and the algorithm version that produced it.
        """
        params: dict[str, Any] = {field: getattr(self, field) for field in self.CHUNKING_FIELDS}
        params["pipeline_version"] = self.CHUNKING_PIPELINE_VERSION
        return params

    def chunking_sha256(self) -> str:
        """Stable identity of a chunking run, stored in `chunks.chunk_config_sha256`.

        Sorted keys and a canonical separator so the digest depends on the values alone
        and not on dict ordering; the Phase 7 chunk-size sweep selects chunk sets by
        this value, so it has to be reproducible across processes and machines.
        """
        canonical = json.dumps(self.chunking_params(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
