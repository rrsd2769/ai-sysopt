"""Schema initialization for the Data Storage subsystem."""

from __future__ import annotations

from sqlalchemy import Engine, inspect
from sqlalchemy.exc import SQLAlchemyError

from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import StorageSchemaError
from aiperfopt.storage.schema import metadata

logger = get_logger(__name__)


def create_all_tables(engine: Engine) -> None:
    """Create every table defined in :mod:`aiperfopt.storage.schema` if absent.

    Safe to call on every application startup: ``MetaData.create_all``
    only issues ``CREATE TABLE IF NOT EXISTS``-equivalent DDL and is a
    no-op for tables that already exist. This is deliberately called from
    :func:`aiperfopt.bootstrap.bootstrap` so every entrypoint gets a ready
    schema without a separate manual migration step (there is no formal
    migration tool in V1 — see the module docstring in
    :mod:`aiperfopt.storage.schema`).
    """
    try:
        metadata.create_all(engine)
    except SQLAlchemyError as exc:
        raise StorageSchemaError(
            "Failed to create database schema", recoverable=False
        ) from exc

    logger.info(
        "Storage schema ready",
        extra={"tables": sorted(metadata.tables.keys())},
    )


def verify_schema(engine: Engine) -> bool:
    """Check that every expected table exists in the database.

    Intended as a startup/health-check assertion distinct from
    :func:`create_all_tables` — useful when connecting to a database file
    that is expected to already be initialized (e.g. a read-only reporting
    process) and where silently creating tables would be the wrong
    behavior.
    """
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    expected = set(metadata.tables.keys())
    missing = expected - existing
    if missing:
        logger.warning("Storage schema incomplete", extra={"missing_tables": sorted(missing)})
        return False
    return True
