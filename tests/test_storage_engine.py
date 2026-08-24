from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from aiperfopt.storage import build_sqlite_url, check_connectivity, create_storage_engine
from aiperfopt.storage.engine import engine_connection


def test_build_sqlite_url_creates_parent_directory(tmp_path: Path):
    db_path = tmp_path / "nested" / "dir" / "aiperfopt.db"
    assert not db_path.parent.exists()

    url = build_sqlite_url(db_path)

    assert db_path.parent.exists()
    assert url.startswith("sqlite:///")
    assert str(db_path) in url


def test_create_storage_engine_applies_pragmas(tmp_path: Path):
    engine = create_storage_engine(tmp_path / "aiperfopt.db")

    with engine_connection(engine) as conn:
        fk = conn.execute(text("PRAGMA foreign_keys")).scalar()
        journal_mode = conn.execute(text("PRAGMA journal_mode")).scalar()
        synchronous = conn.execute(text("PRAGMA synchronous")).scalar()

    assert fk == 1
    assert journal_mode == "wal"
    # NORMAL == 1 per SQLite's PRAGMA synchronous encoding.
    assert synchronous == 1

    engine.dispose()


def test_check_connectivity_true_for_healthy_engine(tmp_path: Path):
    engine = create_storage_engine(tmp_path / "aiperfopt.db")
    assert check_connectivity(engine) is True
    engine.dispose()


def test_check_connectivity_false_for_disposed_engine_pointing_nowhere(tmp_path: Path):
    # A directory that cannot hold a db file (points at a path that is
    # itself a directory) should fail connectivity gracefully rather than
    # raising out of check_connectivity.
    bad_path = tmp_path / "not_a_file"
    bad_path.mkdir()

    engine = create_storage_engine(bad_path)  # engine creation itself succeeds (lazy connect)
    assert check_connectivity(engine) is False
    engine.dispose()
