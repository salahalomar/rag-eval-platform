import pytest
from pydantic import ValidationError

from rag.config import RetrievalConfig


def test_defaults_match_the_specification() -> None:
    # These defaults are the documented baseline arm. If one changes, every committed
    # eval result that omitted it silently means something different, so pin them.
    c = RetrievalConfig()
    assert c.embedding_model == "BAAI/bge-small-en-v1.5"
    assert c.chunk_tokens == 512
    assert c.chunk_overlap_pct == pytest.approx(0.15)
    assert c.drop_references is True
    assert c.drop_figure_only_pages is True
    assert c.contextual_headers is True
    assert c.dense_enabled is True
    assert c.dense_top_k == 50
    assert c.lexical_enabled is True
    assert c.lexical_top_k == 50
    assert c.fusion == "rrf"
    assert c.rrf_k == 60
    assert c.rerank_enabled is True
    assert c.rerank_model == "BAAI/bge-reranker-base"
    assert c.final_top_k == 5
    assert c.score_floor == pytest.approx(0.0)


def test_config_is_frozen() -> None:
    c = RetrievalConfig()
    with pytest.raises(ValidationError):
        c.final_top_k = 10  # type: ignore[misc]


def test_config_is_hashable_and_compares_by_value() -> None:
    # The Phase 7 judgement cache keys on the config, which requires both properties.
    assert RetrievalConfig() == RetrievalConfig()
    assert hash(RetrievalConfig()) == hash(RetrievalConfig())
    assert len({RetrievalConfig(), RetrievalConfig(final_top_k=3)}) == 2


def test_fusion_mode_is_constrained() -> None:
    with pytest.raises(ValidationError):
        RetrievalConfig(fusion="bm25")  # type: ignore[arg-type]


def test_round_trips_through_json() -> None:
    # Every eval result embeds the config; if it cannot round-trip, old results become
    # unreadable the moment a field is added.
    c = RetrievalConfig(chunk_tokens=256, rerank_enabled=False)
    assert RetrievalConfig.model_validate_json(c.model_dump_json()) == c
