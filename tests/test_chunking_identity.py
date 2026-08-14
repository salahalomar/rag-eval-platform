"""The chunking fingerprint that keys stored chunk sets.

Phase 7's chunk-size sweep selects chunks by `chunk_config_sha256`. If two genuinely
different chunkings ever collide under one fingerprint, the sweep silently compares a
configuration against itself and reports a difference of zero -- a result that looks
like a finding.
"""

import hashlib
import json

import pytest

from rag.config import RetrievalConfig


def test_fingerprint_is_stable_across_instances() -> None:
    assert RetrievalConfig().chunking_sha256() == RetrievalConfig().chunking_sha256()


@pytest.mark.parametrize(
    "field,value",
    [
        ("embedding_model", "BAAI/bge-base-en-v1.5"),  # different tokenizer, new boundaries
        ("chunk_tokens", 256),
        ("chunk_overlap_pct", 0.25),
        ("drop_references", False),
        ("drop_figure_only_pages", False),
        ("contextual_headers", False),
    ],
)
def test_every_chunking_field_changes_the_fingerprint(field: str, value: object) -> None:
    base = RetrievalConfig()
    assert base.model_copy(update={field: value}).chunking_sha256() != base.chunking_sha256()


@pytest.mark.parametrize(
    "field,value",
    [
        ("dense_top_k", 25),
        ("lexical_top_k", 25),
        ("fusion", "dense_only"),
        ("rrf_k", 20),
        ("rerank_enabled", False),
        ("final_top_k", 10),
        ("score_floor", 0.5),
    ],
)
def test_search_only_fields_do_not_change_the_fingerprint(field: str, value: object) -> None:
    # Otherwise every retrieval ablation arm would re-chunk and re-embed the entire
    # corpus to obtain identical chunks under a new identity.
    base = RetrievalConfig()
    assert base.model_copy(update={field: value}).chunking_sha256() == base.chunking_sha256()


def test_params_cover_exactly_the_declared_fields() -> None:
    assert set(RetrievalConfig().chunking_params()) == {
        *RetrievalConfig.CHUNKING_FIELDS,
        "pipeline_version",
    }


def test_pipeline_version_participates_in_the_fingerprint() -> None:
    # Config alone cannot express "the code that produced this". Without the version in
    # the digest, a corpus re-ingested after a chunker fix is skipped as unchanged and
    # the database keeps serving chunks built by the old logic.
    config = RetrievalConfig()
    params = config.chunking_params()
    assert params["pipeline_version"] == RetrievalConfig.CHUNKING_PIPELINE_VERSION

    bumped = dict(params, pipeline_version=params["pipeline_version"] + 1)
    canonical = json.dumps(bumped, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() != config.chunking_sha256()


def test_fingerprint_ignores_key_ordering() -> None:
    # Serialised with sorted keys so the digest depends on values alone; a dict-order
    # dependency would make it differ between processes.
    config = RetrievalConfig()
    assert config.chunking_sha256() == RetrievalConfig(**config.model_dump()).chunking_sha256()
