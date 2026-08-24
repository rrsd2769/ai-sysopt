"""Shared helpers for the api/ test modules. Not named ``conftest.py`` so
pytest doesn't auto-discover it as a fixture module for the whole suite --
each api test file imports what it needs explicitly, keeping these
fast-settings/app-building helpers scoped to the tests that actually use
FastAPI's TestClient.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from fastapi.testclient import TestClient

from aiperfopt.api.app import create_app
from aiperfopt.bootstrap import AppContext
from aiperfopt.config import load_settings
from aiperfopt.storage import create_all_tables, create_storage_engine, insert_system_metric_samples_batch
from aiperfopt.storage.records import SystemMetricSample


def build_test_ctx(tmp_path: Path, live_inference_interval_seconds: float = 1.0) -> AppContext:
    settings = load_settings()
    settings = settings.model_copy(update={
        "storage": settings.storage.model_copy(update={"database_path": tmp_path / "test.db"}),
        "forecasting": settings.forecasting.model_copy(update={
            "short_horizon_seconds": (10, 20), "medium_horizon_seconds": (30, 60),
            "walk_forward_splits": 2, "min_training_rows": 15, "models_dir": tmp_path / "models",
        }),
        "retraining": settings.retraining.model_copy(update={"active_version_file": tmp_path / "active.json"}),
        "api": settings.api.model_copy(update={
            "live_inference_interval_seconds": live_inference_interval_seconds
        }),
    })
    engine = create_storage_engine(settings.storage.database_path)
    create_all_tables(engine)
    return AppContext(settings=settings, storage_engine=engine)


def rising_cpu_samples_now(n: int = 300, interval: float = 2.0) -> list[SystemMetricSample]:
    """Real wall-clock timestamps -- required for anything exercising the
    live-inference loop, which reads relative to ``time.time()``, not a
    dataset's own synthetic clock (see PROJECT_HANDOFF.md's Phase 9 notes)."""
    now = time.time()
    return [
        SystemMetricSample(
            timestamp=now - (n - i) * interval,
            cpu_percent=max(0.0, min(95.0, 5.0 + (i / n) * 90.0 + 4.0 * math.sin(i / 3.0))),
            mem_percent=50.0,
        )
        for i in range(n)
    ]


def build_test_client(tmp_path: Path, with_history: bool = False, auto_start: bool = False) -> TestClient:
    ctx = build_test_ctx(tmp_path)
    if with_history:
        insert_system_metric_samples_batch(ctx.storage_engine, rising_cpu_samples_now())
    app = create_app(ctx, auto_start=auto_start)
    return TestClient(app)
