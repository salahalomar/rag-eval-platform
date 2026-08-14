"""Postgres connectivity for the whole library.

Why synchronous rather than async: the library has to be callable identically from a
script, a notebook, the API and the eval runner. An async core would force the eval
runner to wrap every call, and a second way of invoking retrieval is precisely the
thing ENGINEERING.md rules out -- it is how these projects end up publishing metrics
for code that is not the code they ship. FastAPI hands synchronous endpoints to a
threadpool, so nothing is given up by staying sync, and the dominant cost later is a
CPU-bound cross-encoder that blocks either way.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pydantic import BaseModel

from rag.settings import get_settings


class DatabaseHealth(BaseModel):
    """What the process can currently observe about its database."""

    connected: bool
    server_version: str | None = None
    pgvector_version: str | None = None
    applied_migrations: int | None = None
    error: str | None = None


@contextmanager
def connect(dsn: str | None = None) -> Iterator[psycopg.Connection]:
    """Open a connection, committing on clean exit and rolling back on failure.

    Takes an explicit DSN so that tests and the eval runner can point at a scratch
    database without mutating process-wide state; falls back to settings otherwise.
    """
    with psycopg.connect(dsn or get_settings().database_url) as conn:
        yield conn


def check_health(dsn: str | None = None) -> DatabaseHealth:
    """Probe the database without raising, for the /health endpoint.

    Returns a populated record rather than throwing because a health check that 500s
    tells an operator less than one that reports exactly which part is missing.
    `pgvector_version` is null until migration 001 has been applied, which makes the
    endpoint a usable check on whether the stack is actually ready.
    """
    try:
        with connect(dsn) as conn:
            server_version = _scalar(conn, "SHOW server_version")
            pgvector_version = _scalar(
                conn, "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            applied_migrations: int | None = None
            if _table_exists(conn, "schema_migrations"):
                count = _scalar(conn, "SELECT count(*) FROM schema_migrations")
                applied_migrations = int(count) if count is not None else None
            return DatabaseHealth(
                connected=True,
                server_version=server_version,
                pgvector_version=pgvector_version,
                applied_migrations=applied_migrations,
            )
    except psycopg.Error as exc:
        return DatabaseHealth(connected=False, error=_first_line(str(exc)))
    except OSError as exc:  # host unreachable, DNS failure, refused connection
        return DatabaseHealth(connected=False, error=_first_line(str(exc)))


def _scalar(conn: psycopg.Connection, sql: str) -> str | None:
    row = conn.execute(sql).fetchone()
    if row is None or row[0] is None:
        return None
    return str(row[0])


def _table_exists(conn: psycopg.Connection, name: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (name,)).fetchone()
    return bool(row is not None and row[0])


def _first_line(message: str) -> str:
    """Keep health payloads readable; psycopg errors carry multi-line context."""
    return message.strip().splitlines()[0] if message.strip() else "unknown error"
