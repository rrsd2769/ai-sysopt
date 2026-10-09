"""
Overhead budget measurement (the monitor must not be a burden
on a machine it runs on passively for days; first enforced
only by ``tests/test_overhead.py``).

Extracted into a standalone, importable function so there is exactly one
implementation of "spin up an isolated pipeline with realistic settings
and measure its self-CPU and database growth" — the test suite calls it
to enforce the budget as a regression gate, and ``aiperfopt evaluate --report`` calls it to produce ``overhead.txt``'s
numbers for the evidence pack. Both need the *same* measurement,
not two copies that could silently drift apart.

Deliberately measured against an isolated, temporary pipeline/database
(never the live, already-running collector) — this avoids any SQLite
contention with a real multi-hour collection run and keeps the number
reproducible regardless of what else the machine happens to be doing at
report time.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import psutil
from sqlalchemy import text

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
from aiperfopt.monitoring.pipeline import create_monitoring_pipeline
from aiperfopt.storage import create_all_tables, create_storage_engine

# Documented budgets (refined against
# measured steady-state overhead on real hardware -- see
# tests/test_overhead.py's original module docstring for the full
# rationale). Kept here as the single source of truth; the test suite
# imports these rather than redefining them.
CPU_BUDGET_PERCENT = 3.0
DB_GROWTH_CEILING_BYTES_PER_HOUR = 200 * 1024 * 1024  # 200 MB/hour

_WARMUP_SECONDS = 2.0  # excluded from measurement: thread/priming startup cost, not steady-state
_RUN_SECONDS = 8.0


@dataclass(frozen=True)
class OverheadMeasurement:
    cpu_percent: float
    db_growth_bytes_per_hour: float
    cpu_budget_percent: float = CPU_BUDGET_PERCENT
    db_growth_ceiling_bytes_per_hour: float = DB_GROWTH_CEILING_BYTES_PER_HOUR


def _isolated_settings_context(work_dir: Path) -> AppContext:
    """Realistic (not test-accelerated) polling intervals and top-N bound --
    this measures the cost of the *real* configuration, not a fast-polling
    test double."""
    settings = AppSettings(
        monitoring=MonitoringSettings(
            polling_interval_seconds=1.5, process_polling_interval_seconds=3.0,
            process_capture_top_n=20,
        ),
        forecasting=ForecastingSettings(
            short_horizon_seconds=(30, 60), medium_horizon_seconds=(120, 300),
            walk_forward_splits=3, min_training_rows=20,
            models_dir=work_dir / "models", directional_stability_threshold=0.02,
            baseline_improvement_margin=0.05,
        ),
        contention=ContentionSettings(
            decision_threshold=0.75, anomaly_dampening_factor=0.3,
            anomaly_significant_threshold=0.7, contributing_processes_top_n=5,
        ),
        diagnosis=DiagnosisSettings(
            cross_resource_dominance_ratio=0.6, process_dominance_ratio=0.5,
            memory_sudden_rate_of_change_threshold=0.2, shap_top_n=5,
        ),
        privacy=PrivacySettings(excluded_processes=[]),
        storage=StorageSettings(
            database_path=work_dir / "overhead.db", full_resolution_retention_hours=72,
            process_retention_hours=6,
        ),
        logging=LoggingSettings(level="WARNING", log_dir=work_dir / "logs", json_format=True),
        ingestion=IngestionSettings(
            batch_size=200, flush_interval_seconds=5.0, queue_maxsize=20000,
            retention_check_interval_seconds=3600.0,
        ),
    )
    engine = create_storage_engine(settings.storage.database_path)
    create_all_tables(engine)
    return AppContext(settings=settings, storage_engine=engine)


def measure_pipeline_overhead(
    work_dir: Path, warmup_seconds: float = _WARMUP_SECONDS, run_seconds: float = _RUN_SECONDS
) -> OverheadMeasurement:
    """Run an isolated monitoring pipeline for ``run_seconds`` (after
    ``warmup_seconds`` of excluded startup cost) and measure this
    process's own CPU usage and the pipeline's database growth,
    extrapolated to steady-state percent / bytes-per-hour figures.

    ``work_dir`` holds the temporary database/models/logs for the
    isolated run -- never the real, already-running collector's database.
    """
    ctx = _isolated_settings_context(work_dir)
    pipeline = create_monitoring_pipeline(ctx)
    process = psutil.Process(os.getpid())
    db_path = ctx.settings.storage.database_path

    def _checkpointed_size() -> int:
        with ctx.storage_engine.begin() as connection:
            connection.execute(text("PRAGMA wal_checkpoint(FULL)"))
        return db_path.stat().st_size if db_path.exists() else 0

    start_size = _checkpointed_size()
    pipeline.start()
    try:
        time.sleep(warmup_seconds)
        start_cpu_times = process.cpu_times()
        start_wall = time.time()
        time.sleep(run_seconds)
    finally:
        pipeline.stop()

    end_cpu_times = process.cpu_times()
    end_wall = time.time()
    end_size = _checkpointed_size()

    cpu_seconds = (end_cpu_times.user - start_cpu_times.user) + (
        end_cpu_times.system - start_cpu_times.system
    )
    wall_seconds = end_wall - start_wall
    cpu_percent = (cpu_seconds / wall_seconds) * 100.0

    grown_bytes = max(0, end_size - start_size)
    bytes_per_hour = (grown_bytes / wall_seconds) * 3600.0

    return OverheadMeasurement(cpu_percent=cpu_percent, db_growth_bytes_per_hour=bytes_per_hour)
