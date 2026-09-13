from __future__ import annotations

import queue
import time

import psutil
import pytest

from aiperfopt.config.settings import MonitoringSettings
from aiperfopt.monitoring.engine import MonitoringEngine
from aiperfopt.monitoring.queue_items import QueueItem, QueueItemKind


def _fast_settings(**overrides) -> MonitoringSettings:
    base = dict(
        polling_interval_seconds=0.05,
        process_polling_interval_seconds=0.1,
        process_capture_top_n=20,
    )
    base.update(overrides)
    return MonitoringSettings(**base)


@pytest.mark.plumbing
def test_engine_start_stop_lifecycle():
    q: "queue.Queue[QueueItem]" = queue.Queue()
    engine = MonitoringEngine(_fast_settings(), q)

    assert engine.is_running is False
    engine.start()
    assert engine.is_running is True
    engine.stop(timeout=2.0)
    assert engine.is_running is False


@pytest.mark.plumbing
def test_engine_produces_system_metric_items():
    q: "queue.Queue[QueueItem]" = queue.Queue()
    engine = MonitoringEngine(_fast_settings(), q)
    engine.start()
    try:
        item = q.get(timeout=2.0)
        assert item.kind is QueueItemKind.SYSTEM_METRIC
    finally:
        engine.stop(timeout=2.0)


@pytest.mark.plumbing
def test_engine_produces_process_metric_items():
    q: "queue.Queue[QueueItem]" = queue.Queue()
    engine = MonitoringEngine(_fast_settings(), q)
    engine.start()
    try:
        deadline = time.monotonic() + 3.0
        found = False
        while time.monotonic() < deadline:
            try:
                item = q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item.kind is QueueItemKind.PROCESS_METRIC:
                found = True
                break
        assert found
    finally:
        engine.stop(timeout=2.0)


@pytest.mark.plumbing
def test_engine_survives_repeated_system_metric_failures(monkeypatch):
    """FR-1.4 / NFR-5.1: a broken mandatory source must not kill the engine
    thread — it should log and keep retrying next cycle."""
    call_count = {"n": 0}
    real_cpu_percent = psutil.cpu_percent

    def flaky_cpu_percent(interval=None):
        call_count["n"] += 1
        # Let the engine's startup priming call (the very first call)
        # succeed; fail the next three in-loop collection attempts, then
        # recover — isolates the behavior under test (loop resilience)
        # from start()'s own priming call.
        if 1 < call_count["n"] <= 4:
            raise OSError("simulated failure")
        return real_cpu_percent(interval=interval)

    monkeypatch.setattr(psutil, "cpu_percent", flaky_cpu_percent)

    q: "queue.Queue[QueueItem]" = queue.Queue()
    engine = MonitoringEngine(_fast_settings(), q)
    engine.start()
    try:
        # Give it enough cycles to get past the failures and succeed.
        deadline = time.monotonic() + 3.0
        got_system_metric = False
        while time.monotonic() < deadline:
            try:
                item = q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item.kind is QueueItemKind.SYSTEM_METRIC:
                got_system_metric = True
                break
        assert got_system_metric
        assert engine.is_running is True  # thread survived the failures
    finally:
        engine.stop(timeout=2.0)


@pytest.mark.plumbing
def test_engine_drops_samples_without_blocking_when_queue_full():
    tiny_queue: "queue.Queue[QueueItem]" = queue.Queue(maxsize=1)
    engine = MonitoringEngine(_fast_settings(), tiny_queue)
    engine.start()
    try:
        # Let it run long enough to fill the size-1 queue and attempt
        # further puts that must be dropped, not block.
        time.sleep(0.6)
        assert engine.dropped_sample_count > 0
        assert engine.is_running is True  # producer threads never blocked/died
    finally:
        engine.stop(timeout=2.0)


@pytest.mark.plumbing
def test_engine_start_is_idempotent():
    q: "queue.Queue[QueueItem]" = queue.Queue()
    engine = MonitoringEngine(_fast_settings(), q)
    engine.start()
    first_thread = engine._system_thread
    engine.start()  # should be a no-op, not spawn a second thread
    second_thread = engine._system_thread
    assert first_thread is second_thread
    engine.stop(timeout=2.0)


@pytest.mark.behavioural
def test_engine_respects_process_exclusions_end_to_end():
    import subprocess

    q: "queue.Queue[QueueItem]" = queue.Queue()
    engine = MonitoringEngine(
        _fast_settings(), q, excluded_process_names=frozenset({"sleep"})
    )
    engine.start()
    proc = subprocess.Popen(["sleep", "2"])
    try:
        time.sleep(1.0)
        drained = []
        while True:
            try:
                drained.append(q.get_nowait())
            except queue.Empty:
                break
        assert all(
            getattr(item.payload, "pid", None) != proc.pid for item in drained
        ), "excluded process's pid leaked onto the queue"
    finally:
        engine.stop(timeout=2.0)
        proc.terminate()
        proc.wait()
