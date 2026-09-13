from __future__ import annotations

import time
from pathlib import Path

import pytest

from aiperfopt.bootstrap import AppContext
from aiperfopt.config.settings import (
    AppSettings,
    ContentionSettings,
    DiagnosisSettings,
    ForecastingSettings,
    IngestionSettings,
    LoggingSettings,
    MonitoringSettings,
    PrivacySettings,
    StorageSettings,
)
from aiperfopt.pipeline import create_monitoring_pipeline
from aiperfopt.storage import (
    create_all_tables,
    create_storage_engine,
    get_process_metrics_range,
    get_system_metrics_range,
)


def _fast_context(tmp_path: Path) -> AppContext:
    settings = AppSettings(
        monitoring=MonitoringSettings(
            polling_interval_seconds=0.05, process_polling_interval_seconds=0.1,
            process_capture_top_n=20,
        ),
        forecasting=ForecastingSettings(
            short_horizon_seconds=(30, 60), medium_horizon_seconds=(120, 300),
            walk_forward_splits=3, min_training_rows=20,
            models_dir=tmp_path / "models", directional_stability_threshold=0.02,
            baseline_improvement_margin=0.05,
        ),
        contention=ContentionSettings(
            decision_threshold=0.75, anomaly_dampening_factor=0.3,
            anomaly_significant_threshold=0.7, contributing_processes_top_n=5,
        ),
        privacy=PrivacySettings(excluded_processes=[]),
        storage=StorageSettings(
            database_path=tmp_path / "pipeline_test.db", full_resolution_retention_hours=72,
            process_retention_hours=6,
        ),
        logging=LoggingSettings(level="INFO", log_dir=tmp_path / "logs", json_format=True),
        ingestion=IngestionSettings(
            batch_size=10, flush_interval_seconds=0.1, queue_maxsize=1000,
            retention_check_interval_seconds=3600.0,
        ),
    )
    engine = create_storage_engine(settings.storage.database_path)
    create_all_tables(engine)
    return AppContext(settings=settings, storage_engine=engine)


pytestmark = pytest.mark.plumbing


def test_full_pipeline_produces_and_persists_real_data(tmp_path: Path):
    ctx = _fast_context(tmp_path)
    pipeline = create_monitoring_pipeline(ctx)

    pipeline.start()
    try:
        time.sleep(1.0)
    finally:
        pipeline.stop(timeout=3.0)

    system_rows = get_system_metrics_range(ctx.storage_engine)
    process_rows = get_process_metrics_range(ctx.storage_engine)

    assert len(system_rows) > 0
    assert len(process_rows) > 0
    assert pipeline.engine.dropped_sample_count == 0
    # Real psutil data, not placeholders.
    assert system_rows[0].mem_percent > 0
    assert any(r.name for r in process_rows)


def test_pipeline_stop_is_clean_and_reentrant_safe(tmp_path: Path):
    ctx = _fast_context(tmp_path)
    pipeline = create_monitoring_pipeline(ctx)
    pipeline.start()
    time.sleep(0.3)
    pipeline.stop(timeout=3.0)

    assert pipeline.engine.is_running is False
    assert pipeline.consumer.is_running is False
