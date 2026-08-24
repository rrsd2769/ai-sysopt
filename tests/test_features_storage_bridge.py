from __future__ import annotations

import time

from sqlalchemy import Engine

from aiperfopt.features import FeatureEngineeringConfig
from aiperfopt.features.storage_bridge import (
    backfill_feature_matrix,
    compute_and_store_live_feature_vector,
    compute_live_feature_vector,
)
from aiperfopt.storage import (
    SystemMetricSample,
    get_features_range,
    insert_system_metric_samples_batch,
)


def _seed_regular_history(engine: Engine, count: int, interval: float, end_ts: float) -> None:
    start = end_ts - count * interval
    samples = [
        SystemMetricSample(
            timestamp=start + i * interval,
            cpu_percent=30.0 + (i % 5),
            mem_percent=40.0,
            disk_read_bytes=1000 * i,
        )
        for i in range(count)
    ]
    insert_system_metric_samples_batch(engine, samples)


def test_compute_live_feature_vector_reads_correct_window(storage_engine: Engine):
    now = time.time()
    _seed_regular_history(storage_engine, count=100, interval=2.0, end_ts=now)

    config = FeatureEngineeringConfig(lookback_windows_seconds=(30, 120), tolerance_seconds=3.0)
    vector = compute_live_feature_vector(storage_engine, config, as_of=now)

    assert vector is not None
    assert vector.timestamp <= now
    assert "cpu_percent_roll_mean_30s" in vector.feature_values


def test_compute_live_feature_vector_none_when_no_data(storage_engine: Engine):
    config = FeatureEngineeringConfig()
    vector = compute_live_feature_vector(storage_engine, config)
    assert vector is None


def test_compute_and_store_live_feature_vector_persists_to_storage(storage_engine: Engine):
    now = time.time()
    _seed_regular_history(storage_engine, count=100, interval=2.0, end_ts=now)
    config = FeatureEngineeringConfig()

    record = compute_and_store_live_feature_vector(storage_engine, config, as_of=now)

    assert record is not None
    assert record.id is not None

    stored = get_features_range(storage_engine)
    assert len(stored) == 1
    assert stored[0].feature_set_version == "v1"


def test_compute_and_store_live_feature_vector_no_op_when_no_data(storage_engine: Engine):
    config = FeatureEngineeringConfig()
    record = compute_and_store_live_feature_vector(storage_engine, config)
    assert record is None
    assert get_features_range(storage_engine) == []


def test_backfill_feature_matrix_persists_all_rows(storage_engine: Engine):
    now = time.time()
    _seed_regular_history(storage_engine, count=100, interval=2.0, end_ts=now)
    config = FeatureEngineeringConfig()

    count = backfill_feature_matrix(storage_engine, config, start=now - 300, end=now, persist=True)

    assert count == 100
    stored = get_features_range(storage_engine)
    assert len(stored) == 100


def test_backfill_feature_matrix_dry_run_does_not_persist(storage_engine: Engine):
    now = time.time()
    _seed_regular_history(storage_engine, count=50, interval=2.0, end_ts=now)
    config = FeatureEngineeringConfig()

    count = backfill_feature_matrix(storage_engine, config, start=now - 200, end=now, persist=False)

    assert count == 50
    assert get_features_range(storage_engine) == []


def test_backfill_feature_matrix_empty_range_returns_zero(storage_engine: Engine):
    config = FeatureEngineeringConfig()
    count = backfill_feature_matrix(storage_engine, config, start=0.0, end=1.0)
    assert count == 0
