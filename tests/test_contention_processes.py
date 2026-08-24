from __future__ import annotations

from sqlalchemy import Engine

from aiperfopt.contention.processes import get_contributing_processes
from aiperfopt.storage import ProcessMetricSample, insert_process_metric_samples_batch


def test_ranks_top_processes_by_cpu_for_cpu_dominant_metric(storage_engine: Engine):
    insert_process_metric_samples_batch(storage_engine, [
        ProcessMetricSample(timestamp=100.0, pid=1, name="chrome", cpu_percent=80.0, memory_percent=10.0),
        ProcessMetricSample(timestamp=100.0, pid=2, name="finder", cpu_percent=5.0, memory_percent=5.0),
        ProcessMetricSample(timestamp=100.0, pid=3, name="slack", cpu_percent=40.0, memory_percent=8.0),
    ])

    result = get_contributing_processes(storage_engine, as_of=100.0, dominant_metric="cpu_percent", top_n=2)

    assert len(result) == 2
    assert result[0].name == "chrome"
    assert result[0].value == 80.0
    assert result[1].name == "slack"


def test_ranks_by_memory_for_memory_dominant_metric(storage_engine: Engine):
    insert_process_metric_samples_batch(storage_engine, [
        ProcessMetricSample(timestamp=100.0, pid=1, name="chrome", cpu_percent=80.0, memory_percent=10.0),
        ProcessMetricSample(timestamp=100.0, pid=2, name="docker", cpu_percent=5.0, memory_percent=60.0),
    ])

    result = get_contributing_processes(storage_engine, as_of=100.0, dominant_metric="mem_percent", top_n=1)

    assert len(result) == 1
    assert result[0].name == "docker"


def test_uses_latest_sample_per_pid_within_lookback(storage_engine: Engine):
    insert_process_metric_samples_batch(storage_engine, [
        ProcessMetricSample(timestamp=95.0, pid=1, name="chrome", cpu_percent=10.0),
        ProcessMetricSample(timestamp=99.0, pid=1, name="chrome", cpu_percent=90.0),  # more recent
    ])

    result = get_contributing_processes(storage_engine, as_of=100.0, dominant_metric="cpu_percent", top_n=1)

    assert result[0].value == 90.0


def test_no_processes_in_window_returns_empty_list(storage_engine: Engine):
    result = get_contributing_processes(storage_engine, as_of=100.0, dominant_metric="cpu_percent", top_n=5)
    assert result == []


def test_rate_suffixed_metric_name_falls_back_to_cpu(storage_engine: Engine):
    """dominant_metric like 'disk_read_bytes_rate' has no direct
    per-process column (Section 9's process-level metric set doesn't
    include per-process I/O) — falls back to cpu_percent as the broadest
    available "which processes are busy" proxy."""
    insert_process_metric_samples_batch(storage_engine, [
        ProcessMetricSample(timestamp=100.0, pid=1, name="backup_daemon", cpu_percent=30.0),
    ])
    result = get_contributing_processes(
        storage_engine, as_of=100.0, dominant_metric="disk_read_bytes_rate", top_n=1
    )
    assert len(result) == 1
    assert result[0].value == 30.0
