from __future__ import annotations

import psutil
import pytest

from aiperfopt.monitoring.collectors import collect_system_metrics, prime_cpu_percent_counters


def test_collect_system_metrics_returns_populated_sample():
    prime_cpu_percent_counters()
    sample = collect_system_metrics()

    assert sample.timestamp > 0
    assert sample.cpu_percent >= 0.0
    assert 0.0 <= sample.mem_percent <= 100.0
    assert sample.mem_total is not None and sample.mem_total > 0


def test_collect_system_metrics_degrades_gracefully_when_swap_unavailable(monkeypatch):
    def _raise():
        raise OSError("swap unavailable in this sandbox")

    monkeypatch.setattr(psutil, "swap_memory", _raise)

    sample = collect_system_metrics()  # must not raise

    assert sample.swap_total is None
    assert sample.swap_used is None
    assert sample.swap_percent is None
    # Unrelated fields still populated.
    assert sample.mem_percent >= 0.0


def test_collect_system_metrics_degrades_gracefully_when_disk_io_unavailable(monkeypatch):
    monkeypatch.setattr(psutil, "disk_io_counters", lambda: (_ for _ in ()).throw(OSError("no disk io")))

    sample = collect_system_metrics()

    assert sample.disk_read_bytes is None
    assert sample.disk_write_bytes is None


def test_collect_system_metrics_handles_no_battery(monkeypatch):
    monkeypatch.setattr(psutil, "sensors_battery", lambda: None)

    sample = collect_system_metrics()

    assert sample.battery_percent is None
    assert sample.power_plugged is None


def test_collect_system_metrics_propagates_failure_of_mandatory_source(monkeypatch):
    def _raise(interval=None):
        raise OSError("cpu_percent fundamentally broken")

    monkeypatch.setattr(psutil, "cpu_percent", _raise)

    with pytest.raises(OSError):
        collect_system_metrics()


def test_collect_system_metrics_multiple_source_failures_still_returns_sample(monkeypatch):
    monkeypatch.setattr(psutil, "swap_memory", lambda: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(psutil, "disk_io_counters", lambda: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(psutil, "net_io_counters", lambda: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(psutil, "getloadavg", lambda: (_ for _ in ()).throw(OSError()))

    sample = collect_system_metrics()  # must not raise despite four failed sources

    assert sample.swap_total is None
    assert sample.disk_read_bytes is None
    assert sample.net_bytes_sent is None
    assert sample.load_avg_1 is None
    # Mandatory fields still present.
    assert sample.cpu_percent >= 0.0
    assert sample.mem_percent >= 0.0
