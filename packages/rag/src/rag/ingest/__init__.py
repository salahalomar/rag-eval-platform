"""Offline ingestion: arXiv fetch, PDF parse, section detection, structure-aware chunking.

This stage sets the quality ceiling for everything downstream. No reranker recovers a
results table that was split away from its caption, so the chunker is the part of this
package most worth reading carefully.
"""
