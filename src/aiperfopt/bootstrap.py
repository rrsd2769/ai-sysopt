"""
Application bootstrap.

Every entrypoint (future API server, future PySide6 app, CLI tools, test
fixtures) should call :func:`bootstrap` exactly once at startup rather than
constructing config/logging/storage independently. This guarantees the
initialization order that later subsystems depend on:

    1. Load layered configuration (must happen first — logging and
       storage are both configured *from* settings).
    2. Configure structured logging (must happen before storage, so any
       failure constructing the storage engine is logged, not just
       raised).
    3. Create the storage engine and ensure its schema exists
       (``aiperfopt.storage.create_all_tables``).

Returns an :class:`AppContext` bundling the three, which downstream
subsystems (Monitoring, Ingestion, etc. in later phases) take as a
constructor argument rather than reaching for globals — keeping every
subsystem independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine

from aiperfopt.applog import configure_logging, get_logger
from aiperfopt.config import AppSettings, load_settings
from aiperfopt.storage import create_all_tables, create_storage_engine

logger = get_logger(__name__)


@dataclass(frozen=True)
class AppContext:
    """Bundle of fully-initialized, shared application state."""

    settings: AppSettings
    storage_engine: Engine


def bootstrap(user_config_path: Path | str | None = None) -> AppContext:
    """Initialize configuration, logging, and storage, in that order.

    Parameters
    ----------
    user_config_path:
        Optional path to a user override configuration file. See
        :func:`aiperfopt.config.load_settings`.
    """
    settings = load_settings(user_config_path)
    configure_logging(settings.logging)

    logger.info(
        "Starting AI System Performance Optimizer",
        extra={"mode": settings.mode.active_mode.value},
    )

    storage_engine = create_storage_engine(settings.storage.database_path)
    create_all_tables(storage_engine)

    logger.info("Bootstrap complete")
    return AppContext(settings=settings, storage_engine=storage_engine)
