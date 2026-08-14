"""Forward-only SQL migration runner.

Why hand-rolled rather than Alembic: ENGINEERING.md mandates plain numbered SQL with no
ORM autogeneration, and the entire feature set required is "apply files in order,
exactly once, and refuse if history has been rewritten". That is roughly a hundred
lines, and it stays legible to someone reading the repo cold.

The checksum guard is the part worth defending. Without it, "forward-only" is a
convention that a single careless edit to an already-applied file silently breaks --
leaving a developer's database and CI's database with different schemas and no signal
that anything diverged. With it, editing applied history is an error at migrate time.
"""

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg

from rag.db import connect

MIGRATION_PATTERN = re.compile(r"^(?P<version>\d{3})_(?P<name>[a-z0-9_]+)\.sql$")

BOOKKEEPING_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INT PRIMARY KEY,
    name        TEXT NOT NULL,
    checksum    TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class MigrationError(RuntimeError):
    """Base class for problems that should stop a migration run."""


class MigrationDriftError(MigrationError):
    """An already-applied migration file no longer matches what was applied.

    This is the error that makes forward-only real. Recovering means reverting the edit
    and adding a new numbered migration instead.
    """


@dataclass(frozen=True, slots=True)
class Migration:
    """One numbered SQL file on disk."""

    version: int
    name: str
    path: Path
    sql: str

    @property
    def checksum(self) -> str:
        """SHA-256 of the file's contents, used to detect edits to applied history."""
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()

    @property
    def label(self) -> str:
        """Human-facing identifier, e.g. `001_init`."""
        return f"{self.version:03d}_{self.name}"


def discover(migrations_dir: Path) -> list[Migration]:
    """Read every `NNN_name.sql` in `migrations_dir`, ordered by version.

    Rejects duplicate version numbers outright: two files claiming `002` would apply in
    filesystem order, which differs between machines, and a migration order that varies
    by machine is not a migration order.
    """
    if not migrations_dir.is_dir():
        raise MigrationError(f"migrations directory not found: {migrations_dir}")

    migrations: dict[int, Migration] = {}
    for path in sorted(migrations_dir.iterdir()):
        if path.suffix != ".sql":
            continue
        match = MIGRATION_PATTERN.match(path.name)
        if match is None:
            raise MigrationError(
                f"migration filename must look like 001_snake_case.sql, got: {path.name}"
            )
        version = int(match.group("version"))
        if version in migrations:
            raise MigrationError(
                f"duplicate migration version {version:03d}: "
                f"{migrations[version].path.name} and {path.name}"
            )
        migrations[version] = Migration(
            version=version,
            name=match.group("name"),
            path=path,
            sql=path.read_text(encoding="utf-8"),
        )

    return [migrations[v] for v in sorted(migrations)]


def applied_checksums(conn: psycopg.Connection) -> dict[int, str]:
    """Map of already-applied version to the checksum recorded at apply time."""
    rows = conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
    return {int(version): str(checksum) for version, checksum in rows}


def ensure_bookkeeping_table(conn: psycopg.Connection) -> None:
    """Create `schema_migrations` if absent.

    Bootstrapped here rather than in `001_init.sql` because the runner has to be able to
    ask what has been applied before it can apply anything.
    """
    with conn.transaction():
        conn.execute(BOOKKEEPING_DDL)


def pending(migrations: list[Migration], applied: dict[int, str]) -> list[Migration]:
    """Migrations not yet applied, after verifying applied history is unchanged.

    Raises `MigrationDriftError` if a previously applied file has been edited, and
    `MigrationError` if a new migration is numbered below one already applied -- which
    would mean it never runs on any database that is already up to date.
    """
    for migration in migrations:
        recorded = applied.get(migration.version)
        if recorded is not None and recorded != migration.checksum:
            raise MigrationDriftError(
                f"{migration.label} was already applied but its contents have changed. "
                f"Migrations are forward-only: revert {migration.path.name} and add a "
                f"new numbered migration instead."
            )

    outstanding = [m for m in migrations if m.version not in applied]
    if applied and outstanding:
        highest_applied = max(applied)
        out_of_order = [m for m in outstanding if m.version < highest_applied]
        if out_of_order:
            names = ", ".join(m.label for m in out_of_order)
            raise MigrationError(
                f"migration(s) numbered below the highest applied version "
                f"({highest_applied:03d}): {names}. Renumber above it."
            )
    return outstanding


def migrate(conn: psycopg.Connection, migrations_dir: Path) -> list[Migration]:
    """Apply outstanding migrations in order, returning those applied.

    Each migration runs in its own transaction so that a failure halfway through a
    sequence leaves the database at a known version rather than partway through one.
    """
    ensure_bookkeeping_table(conn)
    migrations = discover(migrations_dir)
    outstanding = pending(migrations, applied_checksums(conn))

    for migration in outstanding:
        with conn.transaction():
            conn.execute(migration.sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, checksum) VALUES (%s, %s, %s)",
                (migration.version, migration.name, migration.checksum),
            )
    return outstanding


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: `python -m rag.index.migrate --dir infra/migrations`."""
    parser = argparse.ArgumentParser(description="Apply forward-only SQL migrations.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("infra/migrations"),
        help="directory of numbered .sql files (default: infra/migrations)",
    )
    parser.add_argument("--dsn", default=None, help="override DATABASE_URL")
    parser.add_argument(
        "--status",
        action="store_true",
        help="report applied and pending migrations without applying anything",
    )
    args = parser.parse_args(argv)

    try:
        with connect(args.dsn) as conn:
            if args.status:
                ensure_bookkeeping_table(conn)
                applied = applied_checksums(conn)
                for migration in discover(args.dir):
                    mark = "applied" if migration.version in applied else "pending"
                    print(f"  [{mark:>7}] {migration.label}")
                return 0

            applied_now = migrate(conn, args.dir)
    except MigrationError as exc:
        print(f"migration failed: {exc}", file=sys.stderr)
        return 1
    except psycopg.Error as exc:
        print(f"database error: {exc}", file=sys.stderr)
        return 1

    if not applied_now:
        print("no pending migrations")
    for migration in applied_now:
        print(f"applied {migration.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
