"""Dense retrieval against a live Postgres, using a fake encoder.

The encoder is faked so these tests assert the *retrieval* behaviour -- ordering, tie
breaks, chunking isolation, the no-op re-embed -- rather than the model's opinions, and
so they run in CI without downloading 130MB of weights. Tests that genuinely need the
model carry the `model` marker and live in test_embedding.py.
"""

import hashlib
import math
from collections.abc import Iterator, Sequence
from datetime import date

import psycopg
import pytest

from rag.config import RetrievalConfig
from rag.db import connect
from rag.index.embed import cached_vectors, embed_corpus, pending_chunks, write_vectors
from rag.ingest import store
from rag.ingest.arxiv import PaperMetadata
from rag.ingest.chunk import Chunk
from rag.retrieve import dense

pytestmark = pytest.mark.integration

DIM = 384
PAPER_ID = "0000.99999v1"
MODEL = "BAAI/bge-small-en-v1.5"


class FakeEncoder:
    """Deterministic pseudo-embeddings derived from the text's digest.

    Unit length, so cosine distance behaves as it does in production, and stable across
    processes, so a test that passes once passes again.
    """

    dimension = DIM

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        raw = [(digest[i % len(digest)] - 128) / 128.0 for i in range(DIM)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]


def make_chunk(ordinal: int, text: str) -> Chunk:
    return Chunk(
        ordinal=ordinal,
        section_path=f"{ordinal} Section",
        content=text,
        embed_input=text,
        token_count=len(text.split()),
        page_start=1,
        page_end=1,
        char_start=ordinal * 100,
        char_end=ordinal * 100 + len(text),
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        embed_input_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


@pytest.fixture
def corpus() -> Iterator[tuple[psycopg.Connection, RetrievalConfig, list[Chunk]]]:
    """A tiny isolated corpus, removed afterwards so the dev database is unchanged.

    `chunk_tokens=137` is deliberately a value nothing else uses: it gives this fixture
    its own chunking identity, so counts and cache statistics describe these six chunks
    rather than however many thousand the developer's real corpus happens to hold.
    """
    config = RetrievalConfig(dense_top_k=5, embedding_model=MODEL, chunk_tokens=137)
    chunks = [make_chunk(i, f"chunk number {i} about retrieval") for i in range(6)]
    metadata = PaperMetadata(
        id=PAPER_ID,
        title="Fixture Paper",
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
        try:
            yield conn, config, chunks
        finally:
            conn.execute("DELETE FROM papers WHERE id = %s", (PAPER_ID,))


def embed_fixture(conn: psycopg.Connection, config: RetrievalConfig) -> None:
    embed_corpus(
        conn,
        model=config.embedding_model,
        chunk_config_sha256=config.chunking_sha256(),
        encoder=FakeEncoder(),
    )


def test_search_returns_candidates_ranked_by_descending_score(
    corpus: tuple[psycopg.Connection, RetrievalConfig, list[Chunk]],
) -> None:
    conn, config, _ = corpus
    embed_fixture(conn, config)
    results = dense.search(conn, "chunk number 3 about retrieval", config, encoder=FakeEncoder())

    assert results
    assert [c.rank for c in results] == list(range(1, len(results) + 1))
    assert [c.score for c in results] == sorted((c.score for c in results), reverse=True)


def test_the_exact_text_of_a_chunk_retrieves_that_chunk_first(
    corpus: tuple[psycopg.Connection, RetrievalConfig, list[Chunk]],
) -> None:
    conn, config, chunks = corpus
    embed_fixture(conn, config)
    target = chunks[4]
    top = dense.search(conn, target.content, config, encoder=FakeEncoder())[0]
    assert top.content == target.content
    assert top.score == pytest.approx(1.0, abs=1e-6)


def test_candidates_carry_the_metadata_needed_to_cite_them(
    corpus: tuple[psycopg.Connection, RetrievalConfig, list[Chunk]],
) -> None:
    conn, config, _ = corpus
    embed_fixture(conn, config)
    candidate = dense.search(conn, "chunk number 1", config, encoder=FakeEncoder())[0]
    assert candidate.paper_id == PAPER_ID
    assert candidate.paper_title == "Fixture Paper"
    assert candidate.section_path
    assert candidate.page_start >= 1
    assert candidate.char_end > candidate.char_start


def test_results_are_identical_across_repeated_searches(
    corpus: tuple[psycopg.Connection, RetrievalConfig, list[Chunk]],
) -> None:
    # Two runs of `make eval` on one commit must produce identical retrieval metrics.
    # Unordered ties are the classic way that stops being true, so ordering is pinned by
    # chunk_id after distance.
    conn, config, _ = corpus
    embed_fixture(conn, config)
    first = dense.search(conn, "retrieval", config, encoder=FakeEncoder())
    second = dense.search(conn, "retrieval", config, encoder=FakeEncoder())
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_top_k_is_honoured(
    corpus: tuple[psycopg.Connection, RetrievalConfig, list[Chunk]],
) -> None:
    conn, config, _ = corpus
    embed_fixture(conn, config)
    assert len(dense.search(conn, "retrieval", config, encoder=FakeEncoder(), top_k=2)) == 2


def test_search_finds_nothing_for_a_chunking_that_was_never_embedded(
    corpus: tuple[psycopg.Connection, RetrievalConfig, list[Chunk]],
) -> None:
    # The filter that keeps a chunk-size sweep from mixing arms. It sits inside the index
    # scan rather than after it, so a non-matching chunking returns nothing at all rather
    # than silently returning fewer rows than asked for.
    conn, config, _ = corpus
    embed_fixture(conn, config)
    other = config.model_copy(update={"chunk_tokens": 998})
    assert dense.search(conn, "retrieval", other, encoder=FakeEncoder()) == []


def test_approximate_and_exact_search_agree_on_this_corpus(
    corpus: tuple[psycopg.Connection, RetrievalConfig, list[Chunk]],
) -> None:
    conn, config, _ = corpus
    embed_fixture(conn, config)
    approximate = dense.search(conn, "retrieval", config, encoder=FakeEncoder())
    exact = dense.search(conn, "retrieval", config, encoder=FakeEncoder(), exact=True)
    assert [c.chunk_id for c in approximate] == [c.chunk_id for c in exact]


# --- the embedding cache ----------------------------------------------------


def test_re_embedding_an_unchanged_corpus_is_a_no_op(
    corpus: tuple[psycopg.Connection, RetrievalConfig, list[Chunk]],
) -> None:
    conn, config, chunks = corpus
    first = embed_corpus(
        conn,
        model=MODEL,
        chunk_config_sha256=config.chunking_sha256(),
        encoder=FakeEncoder(),
    )
    assert first.encoded == len(chunks)
    assert first.written == len(chunks)

    second = embed_corpus(
        conn,
        model=MODEL,
        chunk_config_sha256=config.chunking_sha256(),
        encoder=FakeEncoder(),
    )
    assert second.encoded == 0
    assert second.written == 0
    assert second.already_embedded == second.chunks_total == len(chunks)


def test_pending_chunks_empties_once_everything_is_embedded(
    corpus: tuple[psycopg.Connection, RetrievalConfig, list[Chunk]],
) -> None:
    conn, config, chunks = corpus
    assert len(pending_chunks(conn, MODEL, config.chunking_sha256())) == len(chunks)
    embed_fixture(conn, config)
    assert pending_chunks(conn, MODEL, config.chunking_sha256()) == []


def test_vectors_are_reusable_by_embedded_text_digest(
    corpus: tuple[psycopg.Connection, RetrievalConfig, list[Chunk]],
) -> None:
    # The cross-chunking cache: a sweep produces new chunk ids whose embedded text is
    # byte-identical to text already embedded, and re-encoding those is pure waste.
    conn, config, chunks = corpus
    embed_fixture(conn, config)
    digests = [c.embed_input_sha256 for c in chunks]
    cached = cached_vectors(conn, MODEL, digests)
    assert set(cached) == set(digests)
    assert all(len(vector) == DIM for vector in cached.values())


def test_write_vectors_ignores_rows_that_already_exist(
    corpus: tuple[psycopg.Connection, RetrievalConfig, list[Chunk]],
) -> None:
    conn, config, _ = corpus
    embed_fixture(conn, config)
    existing = pending_chunks(conn, MODEL, config.chunking_sha256())
    assert existing == []

    row = conn.execute(
        "SELECT chunk_id, chunk_config_sha256 FROM embeddings_384 WHERE model = %s LIMIT 1",
        (MODEL,),
    ).fetchone()
    assert row is not None
    with conn.transaction():
        written = write_vectors(conn, MODEL, [(int(row[0]), [0.0] * DIM, str(row[1]))])
    assert written == 0


def test_embedding_rows_carry_the_chunking_key(
    corpus: tuple[psycopg.Connection, RetrievalConfig, list[Chunk]],
) -> None:
    # Denormalised from chunks in the same transaction, which is what lets the search
    # filter sit inside the index scan. If it were ever NULL the filter would exclude
    # every row and search would silently return nothing.
    conn, config, _ = corpus
    embed_fixture(conn, config)
    row = conn.execute(
        """
        SELECT count(*) FROM embeddings_384 e
        JOIN chunks c ON c.id = e.chunk_id
        WHERE c.paper_id = %s AND e.chunk_config_sha256 IS DISTINCT FROM c.chunk_config_sha256
        """,
        (PAPER_ID,),
    ).fetchone()
    assert row is not None and row[0] == 0
