"""Integration tests against a live Postgres. Run `make dev` first.

Marked `integration` so the reason for a failure is obvious when the stack is down,
but deliberately not skipped automatically: a health check that passes when there is
no database is worse than no health check.
"""

from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from api.main import app
from rag.db import check_health, connect
from rag.index.migrate import applied_checksums, migrate

pytestmark = pytest.mark.integration

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "infra" / "migrations"


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


def test_health_endpoint_reports_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "db", "version"}
    assert body["status"] == "ok"
    assert body["db"]["connected"] is True
    assert body["db"]["pgvector_version"] is not None
    assert body["version"]
