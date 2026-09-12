from __future__ import annotations

from sqlalchemy import Engine

from aiperfopt.storage import (
    ProcessEvent,
    ProcessMetricAggregate,
    ProcessMetricSample,
    SystemMetricSample,
    apply_retention_policy,
    get_process_events_range,
    get_process_metric_aggregates_range,
    get_process_metrics_range,
    get_rollup_range,
    get_system_metrics_range,
    insert_process_event,
    insert_process_metric_aggregate,
    insert_process_metric_sample,
    insert_system_metric_samples_batch,
)
from aiperfopt.storage.rollup import prune_process_level_data, rollup_system_metrics


def _sample(ts: float, cpu: float, mem: float = 40.0, disk_read: int = 100) -> SystemMetricSample:
    return SystemMetricSample(
        timestamp=ts, cpu_percent=cpu, mem_percent=mem, disk_read_bytes=disk_read
    )


def test_rollup_aggregates_old_rows_into_buckets(storage_engine: Engine):
    # All within the same 3600s bucket starting at 0.
    insert_system_metric_samples_batch(
        storage_engine,
        [_sample(0.0, cpu=10.0, disk_read=100), _sample(100.0, cpu=20.0, disk_read=200),
         _sample(200.0, cpu=30.0, disk_read=300)],
    )

    result = rollup_system_metrics(storage_engine, cutoff=1000.0, bucket_seconds=3600)

    assert result.buckets_written == 1
    assert result.raw_rows_removed == 3

    buckets = get_rollup_range(storage_engine)
    assert len(buckets) == 1
    bucket = buckets[0]
    assert bucket.sample_count == 3
    assert bucket.aggregates["cpu_percent"]["avg"] == 20.0
    assert bucket.aggregates["cpu_percent"]["min"] == 10.0
    assert bucket.aggregates["cpu_percent"]["max"] == 30.0
    assert bucket.aggregates["disk_read_bytes"]["sum"] == 600

    # Source rows should be gone.
    remaining = get_system_metrics_range(storage_engine)
    assert remaining == []


def test_rollup_splits_across_bucket_boundaries(storage_engine: Engine):
    insert_system_metric_samples_batch(
        storage_engine,
        [_sample(0.0, cpu=10.0), _sample(3599.0, cpu=20.0), _sample(3600.0, cpu=30.0)],
    )

    result = rollup_system_metrics(storage_engine, cutoff=7200.0, bucket_seconds=3600)

    assert result.buckets_written == 2
    buckets = sorted(get_rollup_range(storage_engine), key=lambda b: b.bucket_start)
    assert buckets[0].bucket_start == 0.0
    assert buckets[0].sample_count == 2
    assert buckets[1].bucket_start == 3600.0
    assert buckets[1].sample_count == 1


def test_rollup_leaves_recent_rows_untouched(storage_engine: Engine):
    insert_system_metric_samples_batch(
        storage_engine, [_sample(0.0, cpu=10.0), _sample(9000.0, cpu=99.0)]
    )

    # Cutoff only covers the first row.
    result = rollup_system_metrics(storage_engine, cutoff=100.0, bucket_seconds=3600)

    assert result.raw_rows_removed == 1
    remaining = get_system_metrics_range(storage_engine)
    assert len(remaining) == 1
    assert remaining[0].timestamp == 9000.0


def test_rollup_with_no_old_rows_is_a_no_op(storage_engine: Engine):
    insert_system_metric_samples_batch(storage_engine, [_sample(9000.0, cpu=10.0)])

    result = rollup_system_metrics(storage_engine, cutoff=100.0)

    assert result.buckets_written == 0
    assert result.raw_rows_removed == 0
    assert len(get_system_metrics_range(storage_engine)) == 1


def test_prune_process_level_data_removes_only_old_rows(storage_engine: Engine):
    insert_process_metric_sample(
        storage_engine, ProcessMetricSample(timestamp=0.0, pid=1, name="old", cpu_percent=5.0)
    )
    insert_process_metric_sample(
        storage_engine, ProcessMetricSample(timestamp=9000.0, pid=2, name="new", cpu_percent=5.0)
    )
    insert_process_event(
        storage_engine, ProcessEvent(timestamp=0.0, pid=1, name="old", event_type="launch")
    )
    insert_process_event(
        storage_engine, ProcessEvent(timestamp=9000.0, pid=2, name="new", event_type="launch")
    )

    result = prune_process_level_data(storage_engine, cutoff=100.0)

    assert result.process_metric_rows_removed == 1
    assert result.process_event_rows_removed == 1

    remaining_metrics = get_process_metrics_range(storage_engine)
    remaining_events = get_process_events_range(storage_engine)
    assert [r.pid for r in remaining_metrics] == [2]
    assert [r.pid for r in remaining_events] == [2]


def test_apply_retention_policy_end_to_end(storage_engine: Engine):
    now = 100_000.0
    old_ts = now - (80 * 3600)  # 80 hours old
    recent_ts = now - (1 * 3600)  # 1 hour old

    insert_system_metric_samples_batch(
        storage_engine, [_sample(old_ts, cpu=50.0), _sample(recent_ts, cpu=60.0)]
    )
    insert_process_metric_sample(
        storage_engine, ProcessMetricSample(timestamp=old_ts, pid=1, name="old", cpu_percent=5.0)
    )
    insert_process_metric_sample(
        storage_engine, ProcessMetricSample(timestamp=recent_ts, pid=2, name="new", cpu_percent=5.0)
    )

    rollup_result, prune_result = apply_retention_policy(
        storage_engine, full_resolution_retention_hours=72, process_retention_hours=6, as_of=now
    )

    assert rollup_result.raw_rows_removed == 1
    assert prune_result.process_metric_rows_removed == 1

    remaining_system = get_system_metrics_range(storage_engine)
    assert len(remaining_system) == 1
    assert remaining_system[0].timestamp == recent_ts

    rollup_buckets = get_rollup_range(storage_engine)
    assert len(rollup_buckets) == 1


def test_process_retention_is_independent_of_system_retention(storage_engine: Engine):
    """A row too old for the (short) process window but well within the
    (long) system window: system metrics at that timestamp survive while
    process rows at the same timestamp are pruned -- the two retention
    windows are separate cutoffs, not one shared cutoff (Phase 2)."""
    now = 100_000.0
    mid_age_ts = now - (10 * 3600)  # 10 hours old: > 6h process window, < 72h system window

    insert_system_metric_samples_batch(storage_engine, [_sample(mid_age_ts, cpu=42.0)])
    insert_process_metric_sample(
        storage_engine,
        ProcessMetricSample(timestamp=mid_age_ts, pid=7, name="mid", cpu_percent=5.0),
    )
    insert_process_metric_aggregate(
        storage_engine,
        ProcessMetricAggregate(timestamp=mid_age_ts, other_count=3, other_cpu_sum=1.5, other_mem_rss_sum=100),
    )

    rollup_result, prune_result = apply_retention_policy(
        storage_engine, full_resolution_retention_hours=72, process_retention_hours=6, as_of=now
    )

    assert rollup_result.raw_rows_removed == 0  # 10h old is well within the 72h system window
    assert prune_result.process_metric_rows_removed == 1  # but past the 6h process window
    assert prune_result.process_aggregate_rows_removed == 1

    assert len(get_system_metrics_range(storage_engine)) == 1
    assert get_process_metrics_range(storage_engine) == []
    assert get_process_metric_aggregates_range(storage_engine) == []


def test_rollup_result_is_idempotent_on_reduced_source_data(storage_engine: Engine):
    """Re-running rollup after rows are already removed does nothing (no duplicate buckets)."""
    insert_system_metric_samples_batch(storage_engine, [_sample(0.0, cpu=10.0)])

    first = rollup_system_metrics(storage_engine, cutoff=1000.0)
    second = rollup_system_metrics(storage_engine, cutoff=1000.0)

    assert first.buckets_written == 1
    assert second.buckets_written == 0
    assert len(get_rollup_range(storage_engine)) == 1
