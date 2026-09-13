"""Phase 10: `create_app(auto_start=True, start_monitoring=False)` must run
the live-inference loop without also starting the bundled Monitoring
pipeline.

This is what `aiperfopt serve --no-monitor` needs for a replay demo: the
server should read/infer from whatever `aiperfopt replay` is writing into
a replay database, without also polling this real machine and writing its
own samples into that same file concurrently -- the same "replay and live
monitoring must never write to the same database" hazard
`aiperfopt.replay.player.assert_monitoring_not_active` guards against on
the primary database, just in the other direction.

Deliberately does not import `tests/api_conftest.py`'s helpers, to avoid
touching a file shared with the rest of the api/ test suite while its
tests are being re-marked elsewhere -- this builds its own minimal
AppContext directly.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aiperfopt.api.app import create_app
from aiperfopt.bootstrap import AppContext
from aiperfopt.config import load_settings
from aiperfopt.storage import create_all_tables, create_storage_engine

pytestmark = pytest.mark.behavioural


def _build_ctx(tmp_path: Path) -> AppContext:
    settings = load_settings()
    settings = settings.model_copy(update={
        "storage": settings.storage.model_copy(update={"database_path": tmp_path / "test.db"}),
        "api": settings.api.model_copy(update={"live_inference_interval_seconds": 0.2}),
    })
    engine = create_storage_engine(settings.storage.database_path)
    create_all_tables(engine)
    return AppContext(settings=settings, storage_engine=engine)


def test_no_monitor_keeps_live_loop_running_but_not_the_collector(tmp_path):
    ctx = _build_ctx(tmp_path)
    app = create_app(ctx, auto_start=True, start_monitoring=False)

    with TestClient(app):
        time.sleep(0.5)  # give the live-inference thread a couple of ticks
        assert app.state.live_loop.is_running is True
        assert app.state.monitoring_pipeline.engine.is_running is False


def test_default_start_monitoring_still_starts_the_collector(tmp_path):
    ctx = _build_ctx(tmp_path)
    app = create_app(ctx, auto_start=True)  # start_monitoring defaults to True

    with TestClient(app):
        time.sleep(0.5)
        assert app.state.live_loop.is_running is True
        assert app.state.monitoring_pipeline.engine.is_running is True
