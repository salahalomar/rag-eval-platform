import math
from collections.abc import Sequence

import pytest

from rag.index.embed import (
    BGE_QUERY_PREFIX,
    MODEL_DIMENSIONS,
    EmbedReport,
    SentenceTransformerEncoder,
    UnknownModelError,
    dimension_of,
    table_for,
)


class RecordingEncoder(SentenceTransformerEncoder):
    """Captures what would reach the model, without loading one.

    Subclassed rather than reimplemented so the prefix logic under test is the real
    implementation; a reimplementation in the test would prove only that the test agrees
    with itself.
    """

    def __init__(self) -> None:
        self.seen: list[list[str]] = []
        self._dimension = 384

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.seen.append(list(texts))
        return [[0.0] * self._dimension for _ in texts]


# --- the bge prefix asymmetry ----------------------------------------------


def test_queries_get_the_instruction_prefix() -> None:
    encoder = RecordingEncoder()
    encoder.encode_query("how does warmup affect training")
    assert encoder.seen == [[BGE_QUERY_PREFIX + "how does warmup affect training"]]


def test_documents_get_no_prefix() -> None:
    # The failure this guards against is invisible: prefixing passages too still returns
    # plausible neighbours, it just quietly costs several points of recall.
    encoder = RecordingEncoder()
    encoder.encode_documents(["a passage of text", "another passage"])
    assert encoder.seen == [["a passage of text", "another passage"]]
    assert not any(BGE_QUERY_PREFIX in text for batch in encoder.seen for text in batch)


def test_the_same_text_embeds_differently_as_query_and_passage() -> None:
    encoder = RecordingEncoder()
    encoder.encode_query("identical text")
    encoder.encode_documents(["identical text"])
    assert encoder.seen[0] != encoder.seen[1]


# --- the model registry -----------------------------------------------------


@pytest.mark.parametrize(("model", "dim"), sorted(MODEL_DIMENSIONS.items()))
def test_registered_models_resolve_to_their_table(model: str, dim: int) -> None:
    assert dimension_of(model) == dim
    assert table_for(model) == f"embeddings_{dim}"


def test_an_unregistered_model_fails_loudly_and_says_what_to_do() -> None:
    # Failing here costs a second. Failing at the database costs an hour of encoding
    # first, because the dimension mismatch only surfaces on the first INSERT.
    with pytest.raises(UnknownModelError, match="MODEL_DIMENSIONS"):
        dimension_of("some/unregistered-model")


def test_the_two_registered_models_use_different_tables() -> None:
    # The whole point of the per-dimension split: bge-base must not land in a 384-d
    # column, and the Phase 7 ablation must not need a schema change to run.
    assert table_for("BAAI/bge-small-en-v1.5") != table_for("BAAI/bge-base-en-v1.5")


# --- reporting --------------------------------------------------------------


def test_cache_hit_rate_counts_only_vectors_that_were_needed() -> None:
    # already_embedded is excluded on purpose: those vectors were never requested, and
    # counting them would inflate the rate towards 100% on every re-run.
    report = EmbedReport(chunks_total=100, already_embedded=90, cache_hits=4, encoded=6)
    assert report.cache_hit_rate == pytest.approx(0.4)


def test_cache_hit_rate_is_zero_when_nothing_was_needed() -> None:
    assert EmbedReport(chunks_total=10, already_embedded=10).cache_hit_rate == 0.0


def test_report_lines_are_printable() -> None:
    assert any("cache hit rate" in line for line in EmbedReport().as_lines())


# --- normalisation ----------------------------------------------------------


def test_recording_encoder_returns_the_declared_dimension() -> None:
    encoder = RecordingEncoder()
    assert len(encoder.encode_documents(["x"])[0]) == encoder.dimension == 384


def unit_length(vector: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


@pytest.mark.model
def test_real_encoder_produces_normalised_vectors_of_the_right_width() -> None:
    """Requires model weights. Normalisation is what lets cosine and inner product agree."""
    from rag.index.embed import encoder_for

    encoder = encoder_for("BAAI/bge-small-en-v1.5")
    vector = encoder.encode_query("a test query")
    assert len(vector) == 384
    assert unit_length(vector) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.model
def test_real_encoder_is_deterministic_across_calls() -> None:
    """Two runs of `make eval` must agree, and that starts with the embeddings."""
    from rag.index.embed import encoder_for

    encoder = encoder_for("BAAI/bge-small-en-v1.5")
    assert encoder.encode_query("stability check") == encoder.encode_query("stability check")


@pytest.mark.model
def test_real_query_and_passage_embeddings_differ_for_identical_text() -> None:
    """The prefix asymmetry, verified end to end against the actual model."""
    from rag.index.embed import encoder_for

    encoder = encoder_for("BAAI/bge-small-en-v1.5")
    assert encoder.encode_query("identical text") != encoder.encode_documents(["identical text"])[0]
