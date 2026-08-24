from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine

from aiperfopt.applog import reset_logging_for_tests
from aiperfopt.storage import create_all_tables, create_storage_engine


@pytest.fixture(autouse=True)
def _isolated_logging():
    """Ensure each test starts and ends with a clean logging configuration.

    Without this, handlers attached by one test's call to
    ``configure_logging`` would leak into subsequent tests (since Python's
    logging module is process-global state), causing duplicate log lines
    or writes to a since-deleted temp directory.
    """
    reset_logging_for_tests()
    yield
    reset_logging_for_tests()


@pytest.fixture
def storage_engine(tmp_path: Path) -> Engine:
    """A fresh, schema-initialized SQLite engine backed by a temp file.

    Used by every storage reader/writer/rollup test so each test gets an
    isolated database rather than sharing state.
    """
    engine = create_storage_engine(tmp_path / "test.db")
    create_all_tables(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def minimal_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Clear any AIPERFOPT_* variables that might leak in from the host shell.

    Keeps config-layering tests deterministic regardless of the developer's
    or CI runner's local environment.
    """
    for key in list(__import__("os").environ):
        if key.startswith("AIPERFOPT_"):
            monkeypatch.delenv(key, raising=False)
