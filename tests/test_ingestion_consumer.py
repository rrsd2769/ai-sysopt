from __future__ import annotations

import queue
import time

import pytest
from sqlalchemy import Engine

from aiperfopt.config.settings import IngestionSettings
from aiperfopt.ingestion.consumer import IngestionConsumer
from aiperfopt.monitoring.queue_items import QueueItem, QueueItemKind
from aiperfopt.storage import (
    ProcessEvent,
    ProcessMetricSample,
    SystemMetricSample,
    get_process_events_range,
    get_process_metrics_range,
    get_recent_system_metrics,
    get_system_metrics_range,
)


def _settings(**overrides) -> IngestionSettings:
    base = dict(
        batch_size=100,
        flush_interval_seconds=0.15,
        queue_maxsize=1000,
        retention_check_interval_seconds=3600.0,
    )
    base.update(overrides)
    return IngestionSettings(**base)


def _sample(ts: float) -> SystemMetricSample:
    return SystemMetricSample(timestamp=ts, cpu_percent=10.0, mem_percent=20.0)


@pytest.mark.plumbing
def test_consumer_flushes_on_batch_size(storage_engine: Engine):
    q: "queue.Queue[QueueItem]" = queue.Queue()
    consumer = IngestionConsumer(
        storage_engine, q, _settings(batch_size=3, flush_interval_seconds=60.0), full_resolution_retention_hours=72, process_retention_hours=6,
    )
    consumer.start()
    try:
        for i in range(3):
            q.put(QueueItem.system_metric(_sample(float(i))))

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if len(get_system_metrics_range(storage_engine)) >= 3:
                break
            time.sleep(0.05)

        rows = get_system_metrics_range(storage_engine)
        assert len(rows) == 3
    finally:
        consumer.stop(timeout=2.0)


@pytest.mark.plumbing
def test_consumer_flushes_on_interval_with_partial_batch(storage_engine: Engine):
    q: "queue.Queue[QueueItem]" = queue.Queue()
    consumer = IngestionConsumer(
        storage_engine, q, _settings(batch_size=1000, flush_interval_seconds=0.1),
        full_resolution_retention_hours=72, process_retention_hours=6,
    )
    consumer.start()
    try:
        q.put(QueueItem.system_metric(_sample(1.0)))  # far below batch_size

        deadline = time.monotonic() + 2.0
        rows = []
        while time.monotonic() < deadline:
            rows = get_system_metrics_range(storage_engine)
            if rows:
                break
            time.sleep(0.05)

        assert len(rows) == 1
    finally:
        consumer.stop(timeout=2.0)


@pytest.mark.behavioural
def test_consumer_preserves_chronological_order(storage_engine: Engine):
    q: "queue.Queue[QueueItem]" = queue.Queue()
    consumer = IngestionConsumer(
        storage_engine, q, _settings(batch_size=5, flush_interval_seconds=60.0),
        full_resolution_retention_hours=72, process_retention_hours=6,
    )
    consumer.start()
    try:
        timestamps = [5.0, 3.0, 8.0, 1.0, 9.0]  # deliberately produced out of order
        for ts in timestamps:
            q.put(QueueItem.system_metric(_sample(ts)))

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if len(get_system_metrics_range(storage_engine)) >= 5:
                break
            time.sleep(0.05)

        rows = get_system_metrics_range(storage_engine)  # reader always ORDER BY timestamp
        assert [r.timestamp for r in rows] == sorted(timestamps)
    finally:
        consumer.stop(timeout=2.0)


@pytest.mark.plumbing
def test_consumer_routes_all_three_record_types(storage_engine: Engine):
    q: "queue.Queue[QueueItem]" = queue.Queue()
    consumer = IngestionConsumer(
        storage_engine, q, _settings(batch_size=10, flush_interval_seconds=0.1),
        full_resolution_retention_hours=72, process_retention_hours=6,
    )
    consumer.start()
    try:
        q.put(QueueItem.system_metric(_sample(1.0)))
        q.put(QueueItem.process_metric(
            ProcessMetricSample(timestamp=1.0, pid=100, name="test", cpu_percent=5.0)
        ))
        q.put(QueueItem.process_event(
            ProcessEvent(timestamp=1.0, pid=100, name="test", event_type="launch")
        ))

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if (
                get_system_metrics_range(storage_engine)
                and get_process_metrics_range(storage_engine)
                and get_process_events_range(storage_engine)
            ):
                break
            time.sleep(0.05)

        assert len(get_system_metrics_range(storage_engine)) == 1
        assert len(get_process_metrics_range(storage_engine)) == 1
        assert len(get_process_events_range(storage_engine)) == 1
    finally:
        consumer.stop(timeout=2.0)


@pytest.mark.plumbing
def test_consumer_flushes_remaining_items_on_stop(storage_engine: Engine):
    q: "queue.Queue[QueueItem]" = queue.Queue()
    consumer = IngestionConsumer(
        storage_engine, q, _settings(batch_size=1000, flush_interval_seconds=60.0),
        full_resolution_retention_hours=72, process_retention_hours=6,
    )
    consumer.start()
    for i in range(7):
        q.put(QueueItem.system_metric(_sample(float(i))))
    time.sleep(0.05)  # let the consumer dequeue into its in-memory batch
    consumer.stop(timeout=2.0)  # must flush the partial, still-unflushed batch

    rows = get_system_metrics_range(storage_engine)
    assert len(rows) == 7


@pytest.mark.plumbing
def test_consumer_survives_storage_failure_and_keeps_running(storage_engine: Engine, monkeypatch):
    import aiperfopt.ingestion.consumer as consumer_module
    from aiperfopt.common.exceptions import StorageError

    def _always_fail(*args, **kwargs):
        raise StorageError("simulated storage outage")

    monkeypatch.setattr(consumer_module, "insert_system_metric_samples_batch", _always_fail)

    q: "queue.Queue[QueueItem]" = queue.Queue()
    consumer = IngestionConsumer(
        storage_engine, q, _settings(batch_size=2, flush_interval_seconds=60.0),
        full_resolution_retention_hours=72, process_retention_hours=6,
    )
    consumer.start()
    try:
        q.put(QueueItem.system_metric(_sample(1.0)))
        q.put(QueueItem.system_metric(_sample(2.0)))
        time.sleep(0.3)

        assert consumer.is_running is True  # thread survived the storage failure
        assert get_system_metrics_range(storage_engine) == []  # batch was dropped, not written
    finally:
        consumer.stop(timeout=2.0)


@pytest.mark.plumbing
def test_consumer_runs_retention_check_on_schedule(storage_engine: Engine, monkeypatch):
    import aiperfopt.ingestion.consumer as consumer_module

    calls = []

    def _fake_retention(engine, full_resolution_retention_hours, process_retention_hours, as_of=None):
        calls.append((full_resolution_retention_hours, process_retention_hours))
        from aiperfopt.storage.rollup import PruneResult, RollupResult

        return RollupResult(0, 0), PruneResult(0, 0)

    monkeypatch.setattr(consumer_module, "apply_retention_policy", _fake_retention)

    q: "queue.Queue[QueueItem]" = queue.Queue()
    consumer = IngestionConsumer(
        storage_engine, q, _settings(retention_check_interval_seconds=0.1),
        full_resolution_retention_hours=48, process_retention_hours=6,
    )
    consumer.start()
    try:
        time.sleep(0.35)
        assert len(calls) >= 2
        assert calls[0] == (48, 6)
    finally:
        consumer.stop(timeout=2.0)
