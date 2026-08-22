"""Lexical search and the single retrieve() entry point, against a live Postgres.

Both models are faked, so these tests assert routing, fusion and lexical matching
rather than the models' opinions -- and run offline. Reranking is switched off here;
it has its own module.
"""

import hashlib
from collections.abc import Iterator
from datetime import date

import psycopg
import pytest

from conftest import ConstantReranker, FakeEncoder, WordOverlapReranker
from rag.config import RetrievalConfig
from rag.db import connect
from rag.index.embed import embed_corpus
from rag.ingest import store
from rag.ingest.arxiv import PaperMetadata
from rag.ingest.chunk import Chunk
from rag.retrieve import lexical, retrieve
from rag.retrieve.lexical import RANKING_FUNCTION

pytestmark = pytest.mark.integration

PAPER_ID = "0000.88888v1"
MODEL = "BAAI/bge-small-en-v1.5"

# Distinctive vocabulary so lexical matching is unambiguous, plus one near-duplicate pair
# so ranking has something to separate.
TEXTS = [
    "Learning rate warmup stabilises transformer training in the early steps.",
    "The reranker is a cross encoder scoring query and passage together.",
    "Reciprocal rank fusion combines two ranked lists without score normalisation.",
    "Warmup schedules interact with the optimiser and the batch size.",
    "Vector quantisation reduces the memory footprint of an embedding index.",
    "Nothing in this chunk mentions the distinctive terms at all.",
]


def make_chunk(ordinal: int, text: str) -> Chunk:
    return Chunk(
        ordinal=ordinal,
        section_path=f"{ordinal} Section",
        content=text,
        embed_input=text,
        token_count=len(text.split()),
        page_start=1,
        page_end=1,
        char_start=ordinal * 200,
        char_end=ordinal * 200 + len(text),
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        embed_input_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


@pytest.fixture
def corpus() -> Iterator[tuple[psycopg.Connection, RetrievalConfig]]:
    """Six chunks under a chunking identity nothing else uses, removed afterwards."""
    config = RetrievalConfig(
        embedding_model=MODEL,
        chunk_tokens=139,
        rerank_enabled=False,
        dense_top_k=6,
        lexical_top_k=6,
        final_top_k=6,
    )
    chunks = [make_chunk(i, text) for i, text in enumerate(TEXTS)]
    metadata = PaperMetadata(
        id=PAPER_ID,
        title="Lexical Fixture Paper",
        authors=("A Author",),
        abstract="An abstract.",
        categories=("cs.LG",),
        published_at=date(2024, 1, 1),
        pdf_url="https://example.invalid/paper",
    )
    with connect() as conn:
        conn.execute("DELETE FROM papers WHERE id = %s", (PAPER_ID,))
        with conn.transaction():
            store.upsert_paper(conn, metadata, "fixturesha")
            store.insert_chunks(conn, PAPER_ID, chunks, config)
        embed_corpus(
            conn,
            model=MODEL,
            chunk_config_sha256=config.chunking_sha256(),
            encoder=FakeEncoder(),
        )
        try:
            yield conn, config
        finally:
            conn.execute("DELETE FROM papers WHERE id = %s", (PAPER_ID,))


# --- honest naming ----------------------------------------------------------


def test_the_lexical_arm_names_the_function_it_actually_uses() -> None:
    # ts_rank_cd is coverage density: no document-frequency term, no length saturation.
    # Calling it BM25 would be a claim the code does not support, so the name is pinned
    # here and the README quotes this constant.
    assert RANKING_FUNCTION == "ts_rank_cd"


# --- lexical matching -------------------------------------------------------


def test_lexemes_are_stemmed_and_stopwords_removed(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, _ = corpus
    found = lexical.lexemes(conn, "What is the effect of learning rate warmup on training?")
    assert "warmup" in found
    assert "train" in found  # stemmed
    assert "the" not in found and "is" not in found and "what" not in found


def test_lexical_search_finds_the_matching_chunk(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    results = lexical.search(conn, "learning rate warmup", config)
    assert results
    assert "warmup" in results[0].content.lower()


def test_or_semantics_retrieve_more_than_conjunction_would(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    # No single chunk contains every content word of this question. Under plainto_tsquery
    # (which ANDs) the arm would return nothing at all; OR-ing the lexemes is what makes
    # lexical usable as the recall complement to dense.
    conn, config = corpus
    question = "How does learning rate warmup interact with the reranker and fusion?"
    assert len(lexical.search(conn, question, config)) >= 3


def test_more_covered_terms_ranks_higher(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    results = lexical.search(conn, "warmup transformer training stabilises", config)
    assert results[0].content == TEXTS[0]


def test_a_stopword_only_query_returns_nothing_without_erroring(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    # Reached in practice. An exception here would fail a request that dense could
    # still have answered.
    conn, config = corpus
    assert lexical.lexemes(conn, "what about it") == []
    assert lexical.search(conn, "what about it", config) == []


@pytest.mark.parametrize("query", ["", "   ", "???", "!!! ...", "the of and"])
def test_degenerate_queries_are_handled(
    corpus: tuple[psycopg.Connection, RetrievalConfig], query: str
) -> None:
    conn, config = corpus
    assert lexical.search(conn, query, config) == []


def test_lexical_scores_are_bounded_and_descending(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    # Normalisation flag 32 maps rank to r/(r+1). Bounded scores are what let a single
    # score_floor mean anything across arms in Phase 4.
    conn, config = corpus
    results = lexical.search(conn, "warmup training fusion reranker", config)
    assert results
    assert all(0.0 <= c.score < 1.0 for c in results)
    assert [c.score for c in results] == sorted((c.score for c in results), reverse=True)


def test_lexical_results_are_stable_across_runs(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    first = lexical.search(conn, "warmup training", config)
    second = lexical.search(conn, "warmup training", config)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_lexical_respects_the_chunking_filter(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    other = config.model_copy(update={"chunk_tokens": 997})
    assert lexical.search(conn, "warmup", other) == []


# --- the single entry point -------------------------------------------------


def test_all_three_modes_are_reachable_by_config_alone(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    # The acceptance criterion for this phase: an ablation arm is a different config,
    # never a different code path.
    conn, config = corpus
    query = "learning rate warmup transformer"
    for mode in ("dense_only", "lexical_only", "rrf"):
        mode_config = config.model_copy(update={"fusion": mode})
        result = retrieve(query, mode_config, conn, encoder=FakeEncoder())
        assert result.candidates, mode


def test_dense_only_runs_no_lexical_query(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    mode_config = config.model_copy(update={"fusion": "dense_only"})
    result = retrieve("warmup", mode_config, conn, encoder=FakeEncoder())
    assert result.lexical_count == 0
    assert result.dense_count > 0
    assert "lexical_ms" not in result.timings_ms


def test_lexical_only_runs_no_dense_query(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    mode_config = config.model_copy(update={"fusion": "lexical_only"})
    result = retrieve("warmup", mode_config, conn, encoder=FakeEncoder())
    assert result.dense_count == 0
    assert result.lexical_count > 0
    assert "dense_ms" not in result.timings_ms


def test_rrf_runs_both_arms_and_records_both_timings(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    result = retrieve("learning rate warmup", config, conn, encoder=FakeEncoder())
    assert result.dense_count > 0
    assert result.lexical_count > 0
    assert {"dense_ms", "lexical_ms", "fusion_ms"} <= set(result.timings_ms)


def test_fused_output_never_exceeds_the_union_of_the_arms(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    result = retrieve("warmup training fusion", config, conn, encoder=FakeEncoder())
    assert result.fused_count <= result.dense_count + result.lexical_count


def test_final_top_k_truncates_the_result(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    narrow = config.model_copy(update={"final_top_k": 2})
    result = retrieve("warmup", narrow, conn, encoder=FakeEncoder())
    assert len(result) == 2
    assert [c.rank for c in result] == [1, 2]


def test_disabling_an_arm_degrades_rrf_to_the_other(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    # fusion=rrf with one arm off is a monotonic transform of the surviving arm, so it
    # returns that arm's order rather than erroring.
    conn, config = corpus
    without_lexical = config.model_copy(update={"lexical_enabled": False})
    fused = retrieve("warmup training", without_lexical, conn, encoder=FakeEncoder())
    dense_config = config.model_copy(update={"fusion": "dense_only"})
    dense_only = retrieve("warmup training", dense_config, conn, encoder=FakeEncoder())
    assert fused.lexical_count == 0
    assert fused.chunk_ids == dense_only.chunk_ids


def test_retrieval_result_exposes_ranked_chunk_ids(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    # The form the Phase 7 metrics consume.
    conn, config = corpus
    result = retrieve("warmup", config, conn, encoder=FakeEncoder())
    assert result.chunk_ids == [c.chunk_id for c in result.candidates]
    assert len(result) == len(result.chunk_ids)


def test_repeated_retrieval_is_identical(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    first = retrieve("learning rate warmup", config, conn, encoder=FakeEncoder())
    second = retrieve("learning rate warmup", config, conn, encoder=FakeEncoder())
    assert first.chunk_ids == second.chunk_ids
    assert [c.score for c in first] == [c.score for c in second]


def test_a_stopword_only_query_still_returns_dense_results(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    # The reason the lexical arm returns [] instead of raising: the query still has a
    # dense answer, and one arm failing must not take the request down.
    conn, config = corpus
    result = retrieve("what about it", config, conn, encoder=FakeEncoder())
    assert result.lexical_count == 0
    assert result.dense_count > 0
    assert result.candidates


# --- reranking through the entry point --------------------------------------


def test_reranking_reorders_what_retrieve_returns(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    query = "cross encoder reranker passage"
    without = retrieve(query, config, conn, encoder=FakeEncoder())
    with_rerank = retrieve(
        query,
        config.model_copy(update={"rerank_enabled": True, "score_floor": -99.0}),
        conn,
        encoder=FakeEncoder(),
        reranker=WordOverlapReranker(),
    )
    assert with_rerank.rerank_stats is not None
    assert without.rerank_stats is None
    # The chunk that actually contains the query's words wins once a model looks at the
    # pair, which the fused order had no way to know.
    assert "cross encoder" in with_rerank.candidates[0].content.lower()


def test_rerank_stats_ride_along_on_the_result(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    result = retrieve(
        "reranker fusion",
        config.model_copy(update={"rerank_enabled": True, "score_floor": -99.0}),
        conn,
        encoder=FakeEncoder(),
        reranker=WordOverlapReranker(),
    )
    stats = result.rerank_stats
    assert stats is not None
    assert stats.scored > 0
    assert stats.mean_rank_movement >= 0.0


def test_a_top_score_below_the_floor_refuses_with_a_reason(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    # Refusal is a measured behaviour, not an error path. Phase 5 declines to call the
    # LLM on exactly this reason.
    conn, config = corpus
    result = retrieve(
        "warmup",
        config.model_copy(update={"rerank_enabled": True, "score_floor": 0.0}),
        conn,
        encoder=FakeEncoder(),
        reranker=ConstantReranker(-3.0),
    )
    assert result.candidates == ()
    assert result.reason == "below_score_floor"
    assert result.refused


def test_a_top_score_above_the_floor_is_returned(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    result = retrieve(
        "warmup",
        config.model_copy(update={"rerank_enabled": True, "score_floor": 0.0}),
        conn,
        encoder=FakeEncoder(),
        reranker=ConstantReranker(3.0),
    )
    assert result.candidates
    assert result.reason is None
    assert not result.refused


def test_an_arm_without_reranking_never_refuses_on_score(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    # Stated as a test because it is a real limitation rather than an oversight: cosine
    # and ts_rank_cd are on scales no single floor value can serve, so the floor gates
    # only the reranker's judgement.
    conn, config = corpus
    result = retrieve(
        "warmup",
        config.model_copy(update={"rerank_enabled": False, "score_floor": 999.0}),
        conn,
        encoder=FakeEncoder(),
    )
    assert result.candidates
    assert result.reason is None


def test_an_empty_result_from_no_matches_is_not_a_refusal(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    # "The corpus has nothing relevant" and "the query matched nothing" are different
    # failures, and only one of them is the system working correctly.
    conn, config = corpus
    other = config.model_copy(update={"chunk_tokens": 996, "rerank_enabled": True})
    result = retrieve("warmup", other, conn, encoder=FakeEncoder(), reranker=WordOverlapReranker())
    assert result.candidates == ()
    assert result.reason is None
    assert not result.refused
