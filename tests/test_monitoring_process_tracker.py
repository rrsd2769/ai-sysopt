from __future__ import annotations

import subprocess
import time

import psutil
import pytest

from aiperfopt.monitoring.process_tracker import ProcessTracker, bound_process_metrics
from aiperfopt.storage.records import ProcessMetricSample


def _wait_for(predicate, timeout: float = 3.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    result = None
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval)
    return result


@pytest.mark.behavioural
def test_first_poll_establishes_baseline_without_launch_events():
    tracker = ProcessTracker()
    result = tracker.poll()

    assert result.events == []  # no launches on the very first (baseline) poll
    assert tracker.tracked_process_count > 0  # the current machine has processes


@pytest.mark.plumbing
def test_second_poll_returns_metrics_for_tracked_processes():
    tracker = ProcessTracker()
    tracker.poll()
    time.sleep(0.15)
    result = tracker.poll()

    assert len(result.metrics) > 0
    sample = result.metrics[0]
    assert sample.pid > 0
    assert sample.name
    assert sample.cpu_percent >= 0.0


@pytest.mark.behavioural
def test_real_subprocess_launch_is_detected():
    tracker = ProcessTracker()
    tracker.poll()  # baseline

    proc = subprocess.Popen(["sleep", "5"])
    try:
        result = _wait_for(
            lambda: next((e for e in tracker.poll().events if e.pid == proc.pid), None)
        )
        assert result is not None
        assert result.event_type == "launch"
        assert result.name == "sleep"
    finally:
        proc.terminate()
        proc.wait()


@pytest.mark.behavioural
def test_real_subprocess_termination_is_detected():
    tracker = ProcessTracker()
    tracker.poll()

    proc = subprocess.Popen(["sleep", "1"])
    # Observe the launch first so the pid enters the cache.
    _wait_for(lambda: next((e for e in tracker.poll().events if e.pid == proc.pid), None))

    proc.wait()  # process exits naturally

    terminate_event = _wait_for(
        lambda: next(
            (e for e in tracker.poll().events if e.pid == proc.pid and e.event_type == "terminate"),
            None,
        )
    )
    assert terminate_event is not None
    assert terminate_event.name == "sleep"


@pytest.mark.behavioural
def test_excluded_process_produces_no_events_or_metrics():
    tracker = ProcessTracker(excluded_process_names=frozenset({"sleep"}))
    tracker.poll()

    proc = subprocess.Popen(["sleep", "2"])
    try:
        time.sleep(0.3)
        result = tracker.poll()
        assert all(e.pid != proc.pid for e in result.events)
        assert all(m.pid != proc.pid for m in result.metrics)
    finally:
        proc.terminate()
        proc.wait()


@pytest.mark.behavioural
def test_exclusion_matching_is_case_insensitive():
    tracker = ProcessTracker(excluded_process_names=frozenset({"sleep"}))
    tracker.poll()

    # psutil reports the binary name lowercase on Linux ("sleep"); this
    # test exercises the matching logic directly rather than depending on
    # platform-specific casing of a real binary name.
    assert tracker._is_excluded("SLEEP") is True
    assert tracker._is_excluded("Sleep") is True
    assert tracker._is_excluded("unrelated") is False


def _sample(pid: int, cpu: float, mem: int) -> ProcessMetricSample:
    return ProcessMetricSample(timestamp=100.0, pid=pid, name=f"p{pid}", cpu_percent=cpu, memory_rss=mem)


@pytest.mark.behavioural
def test_bound_process_metrics_keeps_everything_under_the_bound():
    metrics = [_sample(i, cpu=float(i), mem=i * 10) for i in range(10)]
    kept, aggregate = bound_process_metrics(metrics, top_n=20, timestamp=100.0)

    assert len(kept) == 10
    assert aggregate.other_count == 0
    assert aggregate.other_cpu_sum == 0.0
    assert aggregate.other_mem_rss_sum == 0
    assert aggregate.timestamp == 100.0


@pytest.mark.behavioural
def test_bound_process_metrics_truncates_and_aggregates_the_long_tail():
    # 50 processes, only the union of top-2-by-cpu and top-2-by-mem should
    # survive individually (a small top_n makes the truncation boundary
    # exercised deterministically without needing 40+ synthetic rows).
    metrics = [_sample(i, cpu=float(i), mem=(49 - i) * 10) for i in range(50)]
    kept, aggregate = bound_process_metrics(metrics, top_n=2, timestamp=100.0)

    kept_pids = {m.pid for m in kept}
    # Top-2 by cpu: pids 49, 48. Top-2 by mem (mem = (49-i)*10, so smallest
    # pid has the largest mem): pids 0, 1.
    assert kept_pids == {49, 48, 0, 1}
    assert len(kept) <= 2 * 2

    all_pids = set(range(50))
    other_pids = all_pids - kept_pids
    expected_other_cpu = sum(float(pid) for pid in other_pids)
    expected_other_mem = sum((49 - pid) * 10 for pid in other_pids)

    assert aggregate.other_count == len(other_pids)
    assert aggregate.other_cpu_sum == expected_other_cpu
    assert aggregate.other_mem_rss_sum == expected_other_mem

    # Reconciliation: kept + aggregate reconstructs the untruncated totals.
    assert sum(m.cpu_percent for m in kept) + aggregate.other_cpu_sum == sum(m.cpu_percent for m in metrics)
    assert sum(m.memory_rss for m in kept) + aggregate.other_mem_rss_sum == sum(
        m.memory_rss for m in metrics
    )
    assert len(kept) + aggregate.other_count == len(metrics)


@pytest.mark.behavioural
def test_bound_process_metrics_caps_at_2n_plus_aggregate_regardless_of_population():
    metrics = [_sample(i, cpu=float(i % 7), mem=(i % 5) * 100) for i in range(500)]
    kept, aggregate = bound_process_metrics(metrics, top_n=20, timestamp=100.0)

    assert len(kept) <= 40
    assert aggregate.other_count == len(metrics) - len(kept)


@pytest.mark.plumbing
def test_second_poll_bounds_metrics_via_settings_top_n():
    """End-to-end through ProcessTracker.poll(): a small capture_top_n
    still yields metrics plus a reconciling aggregate on a real machine."""
    tracker = ProcessTracker(capture_top_n=1)
    tracker.poll()
    time.sleep(0.15)
    result = tracker.poll()

    assert result.aggregate is not None
    assert len(result.metrics) <= 2  # union of top-1-by-cpu and top-1-by-mem


@pytest.mark.plumbing
def test_process_disappearing_mid_poll_does_not_raise(monkeypatch):
    """Simulates the per-process degradation: a pid vanishes between
    being listed and being queried."""
    tracker = ProcessTracker()
    tracker.poll()

    real_process_cls = psutil.Process

    class FlakyProcess(real_process_cls):
        def cpu_percent(self, interval=None):
            raise psutil.NoSuchProcess(self.pid)

    monkeypatch.setattr(psutil, "Process", FlakyProcess)

    # Must not raise even though every process will fail on its second
    # poll's cpu_percent() call.
    tracker._processes.clear()
    tracker.poll()
    result = tracker.poll()
    assert isinstance(result.metrics, list)  # completed without raising
