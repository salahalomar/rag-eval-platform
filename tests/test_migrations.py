from pathlib import Path

import pytest

from rag.index.migrate import (
    Migration,
    MigrationDriftError,
    MigrationError,
    discover,
    pending,
)


def write(directory: Path, name: str, sql: str = "SELECT 1;") -> Path:
    path = directory / name
    path.write_text(sql, encoding="utf-8")
    return path


def test_discovers_in_numeric_order(tmp_path: Path) -> None:
    # Written out of order, and 010 sorts before 002 lexically in some locales.
    write(tmp_path, "010_later.sql")
    write(tmp_path, "002_second.sql")
    write(tmp_path, "001_init.sql")
    assert [m.version for m in discover(tmp_path)] == [1, 2, 10]
    assert [m.label for m in discover(tmp_path)] == ["001_init", "002_second", "010_later"]


def test_ignores_non_sql_files(tmp_path: Path) -> None:
    write(tmp_path, "001_init.sql")
    (tmp_path / "README.md").write_text("notes", encoding="utf-8")
    assert len(discover(tmp_path)) == 1


def test_rejects_unparseable_filenames(tmp_path: Path) -> None:
    write(tmp_path, "init.sql")
    with pytest.raises(MigrationError, match="must look like"):
        discover(tmp_path)


def test_rejects_duplicate_versions(tmp_path: Path) -> None:
    # Two files claiming 002 would apply in filesystem order, which varies by machine.
    write(tmp_path, "002_add_chunks.sql")
    write(tmp_path, "002_add_papers.sql")
    with pytest.raises(MigrationError, match="duplicate migration version 002"):
        discover(tmp_path)


def test_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(MigrationError, match="not found"):
        discover(tmp_path / "nope")


def test_checksum_changes_with_contents(tmp_path: Path) -> None:
    path = write(tmp_path, "001_init.sql", "CREATE EXTENSION vector;")
    first = discover(tmp_path)[0].checksum
    path.write_text("CREATE EXTENSION vector; -- edited", encoding="utf-8")
    assert discover(tmp_path)[0].checksum != first


def test_pending_excludes_already_applied(tmp_path: Path) -> None:
    write(tmp_path, "001_init.sql")
    write(tmp_path, "002_core.sql")
    migrations = discover(tmp_path)
    applied = {migrations[0].version: migrations[0].checksum}
    assert [m.label for m in pending(migrations, applied)] == ["002_core"]


def test_pending_is_empty_when_everything_is_applied(tmp_path: Path) -> None:
    write(tmp_path, "001_init.sql")
    migrations = discover(tmp_path)
    assert pending(migrations, {m.version: m.checksum for m in migrations}) == []


def test_editing_applied_history_is_rejected(tmp_path: Path) -> None:
    # The guard that turns "forward-only" from a convention into something enforced.
    write(tmp_path, "001_init.sql", "CREATE EXTENSION vector;")
    migrations = discover(tmp_path)
    with pytest.raises(MigrationDriftError, match="forward-only"):
        pending(migrations, {1: "a-checksum-from-a-previous-version-of-the-file"})


def test_new_migration_numbered_below_applied_head_is_rejected(tmp_path: Path) -> None:
    # 002 would never run on a database already at 003, so fail loudly at migrate time
    # rather than silently on one machine and not another.
    write(tmp_path, "001_init.sql")
    write(tmp_path, "002_squeezed_in.sql")
    write(tmp_path, "003_core.sql")
    migrations = discover(tmp_path)
    applied = {1: migrations[0].checksum, 3: migrations[2].checksum}
    with pytest.raises(MigrationError, match="below the highest applied version"):
        pending(migrations, applied)


def test_migration_label_is_zero_padded() -> None:
    m = Migration(version=7, name="add_index", path=Path("007_add_index.sql"), sql="")
    assert m.label == "007_add_index"


def test_committed_migrations_are_discoverable() -> None:
    # Guards against a real migration file being named in a way the runner rejects.
    repo_root = Path(__file__).resolve().parents[1]
    migrations = discover(repo_root / "infra" / "migrations")
    assert [m.label for m in migrations] == ["001_init"]
