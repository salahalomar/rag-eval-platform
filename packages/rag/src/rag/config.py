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
    # HNSW's per-query search breadth. Not in the original specification, but it changes
    # which chunks come back -- it trades approximate-search recall against latency --
    # and nothing that changes retrieval output may live outside this model, or the
    # result it produced cannot be reconstructed from its record.
    #
    # 400 is measured, not guessed. Against exact-scan ground truth over 50 queries on
    # the 6,386-chunk corpus (`rag bench-index`):
    #
    #     ef_search   recall@10   recall@50   p50 ms   p95 ms
    #            40       0.830       0.860      6.9     11.6
    #           100       0.952       0.921      5.1      8.2
    #           200       0.980       0.973      6.7     12.4
    #           400       1.000       1.000      7.1      8.2
    #           800       1.000       1.000      7.4      8.0
    #
    # Chosen because approximation error here is indistinguishable from retrieval error
    # in the published metrics: at ef_search=100 roughly one true neighbour in twenty is
    # missed, and every ablation arm would carry that deficit while appearing to be a
    # property of the retrieval method. 400 removes it for 2ms.
    #
    # This value is tuned to a corpus of this size and must be re-measured if the corpus
    # grows; 400 will not stay exact at ten times the rows.
    hnsw_ef_search: int = 400
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
