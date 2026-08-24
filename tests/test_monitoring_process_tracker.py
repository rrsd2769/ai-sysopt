from __future__ import annotations

import subprocess
import time

import psutil
import pytest

from aiperfopt.monitoring.process_tracker import ProcessTracker


def _wait_for(predicate, timeout: float = 3.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    result = None
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval)
    return result


def test_first_poll_establishes_baseline_without_launch_events():
    tracker = ProcessTracker()
    result = tracker.poll()

    assert result.events == []  # no launches on the very first (baseline) poll
    assert tracker.tracked_process_count > 0  # the current machine has processes


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


def test_exclusion_matching_is_case_insensitive():
    tracker = ProcessTracker(excluded_process_names=frozenset({"sleep"}))
    tracker.poll()

    # psutil reports the binary name lowercase on Linux ("sleep"); this
    # test exercises the matching logic directly rather than depending on
    # platform-specific casing of a real binary name.
    assert tracker._is_excluded("SLEEP") is True
    assert tracker._is_excluded("Sleep") is True
    assert tracker._is_excluded("unrelated") is False


def test_process_disappearing_mid_poll_does_not_raise(monkeypatch):
    """Simulates FR-1.4's per-process degradation: a pid vanishes between
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
