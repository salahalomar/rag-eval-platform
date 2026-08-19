"""Integration tests against a live Postgres. Run `make dev` first.

Marked `integration` so the reason for a failure is obvious when the stack is down,
but deliberately not skipped automatically: a health check that passes when there is
no database is worse than no health check.
"""

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from api.main import app
from rag.config import RetrievalConfig
from rag.db import check_health, connect
from rag.index.migrate import applied_checksums, migrate
from rag.ingest import store
from rag.ingest.arxiv import PaperMetadata
from rag.ingest.chunk import chunk_document
from rag.ingest.parse import parse_pdf
from rag.ingest.sections import detect_sections
from rag.ingest.tokenization import WhitespaceTokenCounter

pytestmark = pytest.mark.integration

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "infra" / "migrations"
FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


@pytest.fixture(scope="module")
def conn() -> Iterator[psycopg.Connection]:
    with connect() as connection:
        yield connection


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def test_database_is_reachable() -> None:
    health = check_health()
    assert health.connected, health.error
    assert health.server_version is not None
    assert health.server_version.startswith("16.")


def test_migrations_are_applied_and_idempotent(conn: psycopg.Connection) -> None:
    migrate(conn, MIGRATIONS_DIR)  # bring up to head; no-op if already there
    assert migrate(conn, MIGRATIONS_DIR) == [], "second run must apply nothing"
    assert 1 in applied_checksums(conn)


def test_pgvector_is_installed(conn: psycopg.Connection) -> None:
    migrate(conn, MIGRATIONS_DIR)
    row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    assert row is not None, "migration 001 should have created the vector extension"


def test_vector_type_actually_works(conn: psycopg.Connection) -> None:
    # Asserting the extension row exists proves less than exercising the type; a wrong
    # image tag can leave a stale extension registered.
    migrate(conn, MIGRATIONS_DIR)
    row = conn.execute("SELECT '[1,0,0]'::vector <=> '[0,1,0]'::vector").fetchone()
    assert row is not None
    assert float(row[0]) == pytest.approx(1.0)


def test_a_transaction_block_commits_before_the_connection_closes() -> None:
    """Guards the property that makes a long ingest resumable.

    Without autocommit, psycopg opens an implicit transaction on the first statement and
    every later `conn.transaction()` degrades to a savepoint inside it -- so nothing is
    durable until the connection closes, and an interrupted 150-paper ingest loses
    everything. Asserted from a second connection, because the writing connection can
    see its own uncommitted rows and would report success either way.
    """
    with connect() as writer:
        assert writer.autocommit, "connect() must open in autocommit mode"
        writer.execute("DROP TABLE IF EXISTS commit_probe")
        with writer.transaction():
            writer.execute("CREATE TABLE commit_probe (id INT)")
            writer.execute("INSERT INTO commit_probe VALUES (1)")

        # Still inside the writer's `with connect()` block.
        with connect() as reader:
            row = reader.execute("SELECT count(*) FROM commit_probe").fetchone()
            assert row is not None and row[0] == 1

        writer.execute("DROP TABLE commit_probe")


def test_core_schema_tables_exist(conn: psycopg.Connection) -> None:
    migrate(conn, MIGRATIONS_DIR)
    # `embeddings` became `embeddings_384` in migration 003, alongside `embeddings_768`:
    # pgvector fixes dimensionality per column, so a second embedding model needs its own
    # table rather than a nullable second column.
    for table in ("papers", "chunks", "embeddings_384", "embeddings_768", "query_logs"):
        assert conn.execute("SELECT to_regclass(%s)", (table,)).fetchone() != (None,)


def test_reinserting_the_same_chunks_inserts_nothing(conn: psycopg.Connection) -> None:
    """The Phase 1 acceptance criterion, exercised against the real unique constraint."""
    migrate(conn, MIGRATIONS_DIR)
    config = RetrievalConfig(chunk_tokens=40)
    document = parse_pdf(FIXTURE_PDF)
    chunks = chunk_document(
        document,
        detect_sections(document),
        paper_title="Fixture Paper",
        config=config,
        counter=WhitespaceTokenCounter(),
    )
    metadata = PaperMetadata(
        id="0000.00000v1",
        title="Fixture Paper",
        authors=("Test Author",),
        abstract="A fixture.",
        categories=("cs.LG",),
        published_at=date(2024, 1, 1),
        pdf_url="https://example.invalid/0000.00000v1",
    )

    # Rolled back at the end so the test leaves no trace in the shared dev database.
    # psycopg.Rollback is swallowed by the transaction block rather than propagating.
    with conn.transaction():
        store.upsert_paper(conn, metadata, "deadbeef")
        first = store.insert_chunks(conn, metadata.id, chunks, config)
        second = store.insert_chunks(conn, metadata.id, chunks, config)
        assert first == len(chunks) > 0
        assert second == 0, "a second ingest of unchanged content must insert nothing"

        assert store.chunks_exist(conn, metadata.id, config.chunking_sha256())
        # A different chunking is a different experiment and must coexist, not collide.
        other = config.model_copy(update={"chunk_tokens": 80})
        assert not store.chunks_exist(conn, metadata.id, other.chunking_sha256())
        raise psycopg.Rollback


def test_health_endpoint_reports_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "db", "version"}
    assert body["status"] == "ok"
    assert body["db"]["connected"] is True
    assert body["db"]["pgvector_version"] is not None
    assert body["version"]
