"""
SQLite engine bootstrap via SQLAlchemy Core.

This module owns only *connection* concerns — creating the engine,
ensuring the data directory exists, and applying the SQLite pragmas the
rest of the Data Storage subsystem depends on. Table definitions
(``raw_metrics``, ``features``, ``predictions``, ``evaluations``, ``logs``) are deliberately left to the schema and rollup modules.

Rationale for SQLite pragmas applied here:

- ``foreign_keys = ON``: SQLite does not enforce foreign keys by default;
  the schema relies on them for referential integrity
  between raw samples, features, and predictions.
- ``journal_mode = WAL``: allows concurrent readers (e.g. a REST history
  query) while the Ingestion subsystem is writing, which matters because
  ingestion writes must never block the Monitoring Engine,
  and prediction may proceed against live in-memory data if
  storage is briefly locked — WAL substantially reduces how often that
  fallback path is needed.
- ``synchronous = NORMAL``: the standard, safe pairing with WAL — full
  durability on the WAL file with better write throughput than FULL,
  appropriate for a local single-machine app with no distributed
  durability requirement.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import Connection, Engine, create_engine, event, text
from sqlalchemy.exc import SQLAlchemyError

from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import StorageConnectionError

logger = get_logger(__name__)


def build_sqlite_url(database_path: Path) -> str:
    """Build a SQLAlchemy SQLite URL, creating the parent directory if needed."""
    resolved = database_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{resolved}"


def create_storage_engine(database_path: Path, *, echo: bool = False) -> Engine:
    """Create and configure the SQLAlchemy engine used throughout Storage.

    Parameters
    ----------
    database_path:
        Filesystem path to the SQLite database file (from
        ``AppSettings.storage.database_path``).
    echo:
        Passed through to SQLAlchemy's ``echo`` — logs every emitted SQL
        statement. Intended for local debugging only; never enabled by
        default since it would violate the low-overhead requirement
        under sustained polling-driven writes.

    Raises
    ------
    StorageConnectionError
        If the engine cannot be created (e.g. an inaccessible path).
    """
    try:
        url = build_sqlite_url(database_path)
        engine = create_engine(url, echo=echo, future=True)
    except (SQLAlchemyError, OSError) as exc:
        raise StorageConnectionError(
            "Failed to create storage engine",
            recoverable=False,
            context={"database_path": str(database_path)},
        ) from exc

    @event.listens_for(engine, "connect")
    def _apply_sqlite_pragmas(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.execute("PRAGMA synchronous = NORMAL")
        finally:
            cursor.close()

    logger.info("Storage engine created", extra={"database_path": str(database_path)})
    return engine


@contextmanager
def engine_connection(engine: Engine) -> Iterator[Connection]:
    """Context manager yielding a connection, wrapping failures as StorageConnectionError.

    Every direct SQL interaction elsewhere in the Storage subsystem should
    go through this rather than calling ``engine.connect()`` directly, so
    connection-level failures are consistently typed for callers that need
    to implement graceful degradation.
    """
    try:
        with engine.connect() as connection:
            yield connection
    except SQLAlchemyError as exc:
        raise StorageConnectionError(
            "Storage connection failed", context={"error": str(exc)}
        ) from exc


def check_connectivity(engine: Engine) -> bool:
    """Lightweight liveness check (e.g. for a health endpoint or startup check).

    Returns False rather than raising on failure, since this is meant to
    be a cheap probe used in graceful-degradation decision points (e.g.
    e.g. fall back to in-memory-only prediction if storage is
    unavailable), not a hard assertion.
    """
    try:
        with engine_connection(engine) as connection:
            connection.execute(text("SELECT 1"))
        return True
    except StorageConnectionError:
        logger.warning("Storage connectivity check failed", exc_info=True)
        return False
