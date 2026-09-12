from __future__ import annotations

import time

from sqlalchemy import Engine

from aiperfopt.features import FeatureEngineeringConfig
from aiperfopt.features.storage_bridge import (
    backfill_feature_matrix,
    compute_and_store_live_feature_vector,
    compute_live_feature_vector,
    read_process_summary_series,
    summarize_and_persist_process_features,
)
from aiperfopt.storage import (
    ProcessEvent,
    ProcessMetricAggregate,
    ProcessMetricSample,
    SystemMetricSample,
    get_features_range,
    get_process_feature_summaries_range,
    insert_process_event,
    insert_process_metric_aggregate,
    insert_process_metric_sample,
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


def test_summarize_and_persist_process_features_writes_durable_rows(storage_engine: Engine):
    old_ts = 100.0  # older than the cutoff -> about to be pruned
    insert_process_metric_sample(
        storage_engine,
        ProcessMetricSample(timestamp=old_ts, pid=1, name="a", cpu_percent=20.0, memory_rss=1000),
    )
    insert_process_event(
        storage_engine, ProcessEvent(timestamp=old_ts - 5, pid=2, name="b", event_type="launch")
    )
    insert_process_metric_aggregate(
        storage_engine,
        ProcessMetricAggregate(timestamp=old_ts, other_count=3, other_cpu_sum=6.0, other_mem_rss_sum=300),
    )

    written = summarize_and_persist_process_features(storage_engine, cutoff=200.0)

    assert written == 1
    summaries = get_process_feature_summaries_range(storage_engine)
    assert len(summaries) == 1
    assert summaries[0].timestamp == old_ts
    assert summaries[0].summary_values["proc_cpu_sum"] == 20.0 + 6.0
    assert summaries[0].summary_values["proc_launch_count_30s"] == 1


def test_summarize_and_persist_process_features_ignores_rows_at_or_after_cutoff(storage_engine: Engine):
    insert_process_metric_sample(
        storage_engine,
        ProcessMetricSample(timestamp=200.0, pid=1, name="a", cpu_percent=20.0),
    )

    written = summarize_and_persist_process_features(storage_engine, cutoff=200.0)  # strict <

    assert written == 0
    assert get_process_feature_summaries_range(storage_engine) == []


def test_summarize_and_persist_process_features_is_a_noop_with_no_process_data(storage_engine: Engine):
    written = summarize_and_persist_process_features(storage_engine, cutoff=200.0)
    assert written == 0
    assert get_process_feature_summaries_range(storage_engine) == []


def test_compute_and_store_live_feature_vector_persists_to_storage(storage_engine: Engine):
    now = time.time()
    _seed_regular_history(storage_engine, count=100, interval=2.0, end_ts=now)
    config = FeatureEngineeringConfig()

    record = compute_and_store_live_feature_vector(storage_engine, config, as_of=now)

    assert record is not None
    assert record.id is not None

    stored = get_features_range(storage_engine)
    assert len(stored) == 1
    assert stored[0].feature_set_version == "v2"


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


def _seed_process_history(engine: Engine, base: float, poll_interval: float, count: int) -> None:
    for i in range(count):
        ts = base + i * poll_interval
        insert_process_metric_sample(
            engine,
            ProcessMetricSample(timestamp=ts, pid=1, name="worker", cpu_percent=30.0 + i, memory_rss=1000),
        )


def test_read_process_summary_series_combines_persisted_and_raw(storage_engine: Engine):
    """Phase 2's retention split means old data survives only as a
    persisted ProcessFeatureSummaryRecord while recent data is still raw
    -- read_process_summary_series must transparently cover both."""
    old_ts, recent_ts = 100.0, 900.0

    # Old poll: already summarized-and-pruned (Phase 2's steady state) --
    # simulate directly via the persisted summaries table.
    insert_process_metric_sample(
        storage_engine, ProcessMetricSample(timestamp=old_ts, pid=1, name="a", cpu_percent=20.0)
    )
    summarize_and_persist_process_features(storage_engine, cutoff=old_ts + 1)  # prunes the raw row

    # Recent poll: still raw, not yet summarized.
    insert_process_metric_sample(
        storage_engine, ProcessMetricSample(timestamp=recent_ts, pid=2, name="b", cpu_percent=55.0)
    )

    combined = read_process_summary_series(storage_engine, start=0.0, end=1000.0)

    assert sorted(combined["timestamp"].tolist()) == [old_ts, recent_ts]
    old_row = combined[combined["timestamp"] == old_ts].iloc[0]
    recent_row = combined[combined["timestamp"] == recent_ts].iloc[0]
    assert old_row["proc_cpu_sum"] == 20.0
    assert recent_row["proc_cpu_sum"] == 55.0


def test_live_and_backfill_produce_identical_process_columns_for_same_window(storage_engine: Engine):
    """The success criterion this phase is built around: process summary
    columns must be identical whether computed via the live path or the
    batch/training path, for the same window -- asserted directly."""
    now = time.time()
    _seed_regular_history(storage_engine, count=60, interval=2.0, end_ts=now)
    _seed_process_history(storage_engine, base=now - 118.0, poll_interval=5.0, count=24)

    config = FeatureEngineeringConfig(lookback_windows_seconds=(30, 120), tolerance_seconds=3.0)

    live_vector = compute_live_feature_vector(storage_engine, config, as_of=now)
    backfill_count = backfill_feature_matrix(storage_engine, config, start=now - 300, end=now, persist=True)

    assert live_vector is not None
    assert backfill_count > 0
    stored = get_features_range(storage_engine, start=live_vector.timestamp, end=live_vector.timestamp)
    assert len(stored) == 1
    backfilled_row = stored[0]

    process_columns = [c for c in live_vector.feature_values if c.startswith("proc_")]
    assert process_columns, "expected at least one process-derived feature to be present"
    for col in process_columns:
        assert live_vector.feature_values[col] == backfilled_row.feature_values[col], col


def test_compute_live_feature_vector_includes_process_columns_when_available(storage_engine: Engine):
    now = time.time()
    _seed_regular_history(storage_engine, count=60, interval=2.0, end_ts=now)
    _seed_process_history(storage_engine, base=now - 60.0, poll_interval=5.0, count=13)

    config = FeatureEngineeringConfig(lookback_windows_seconds=(30,), tolerance_seconds=3.0)
    vector = compute_live_feature_vector(storage_engine, config, as_of=now)

    assert vector is not None
    assert "proc_cpu_sum_roll_mean_30s" in vector.feature_values


def test_compute_live_feature_vector_has_no_process_columns_without_process_data(storage_engine: Engine):
    now = time.time()
    _seed_regular_history(storage_engine, count=60, interval=2.0, end_ts=now)
    config = FeatureEngineeringConfig(lookback_windows_seconds=(30,), tolerance_seconds=3.0)

    vector = compute_live_feature_vector(storage_engine, config, as_of=now)

    assert vector is not None
    assert not any(k.startswith("proc_") for k in vector.feature_values)
