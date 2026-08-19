"""Retrieval: dense, lexical, fusion and reranking.

Phase 2 ships the dense arm only. Phase 3 adds `retrieve(query, config, conn)` here as
the single entry point that the API and the eval runner both call, and nothing else.
"""

from rag.retrieve.types import Candidate

__all__ = ["Candidate"]
