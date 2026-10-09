"""
System-wide metric collection with per-source graceful degradation.

Each individual metric source (CPU, memory, disk I/O, network I/O, load
average, battery) is collected behind :func:`_safe_source`, which catches
that source's failure, logs it, and leaves the corresponding field(s) as
``None`` rather than aborting the whole poll cycle. A single flaky source
(e.g. a sandboxed environment where ``disk_io_counters`` raises) therefore
degrades one column in one row instead of stopping monitoring entirely.

``psutil.cpu_percent()`` (and, per process, ``Process.cpu_percent()``)
report usage *since the previous call*, and the very first call in a
process's lifetime returns a meaningless ``0.0`` (or the process's
lifetime-average on some platforms). :func:`prime_cpu_percent_counters`
must be called once before the first real sample is collected so that the
first sample enqueued for storage/prediction carries a meaningful value
rather than a spurious 0.0.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

import psutil

from aiperfopt.applog import get_logger
from aiperfopt.storage.records import SystemMetricSample

logger = get_logger(__name__)

T = TypeVar("T")


def prime_cpu_percent_counters() -> None:
    """Seed psutil's internal CPU-percent reference point.

    Call exactly once, before the Monitoring Engine's polling loop starts.
    Without this, the first collected sample's ``cpu_percent`` would be a
    meaningless 0.0 (see module docstring).
    """
    psutil.cpu_percent(interval=None)


def _safe_source(source_name: str, fn: Callable[[], T]) -> T | None:
    """Run a single metric-source call, returning None and logging on failure.

    Implements the "continue operating in a degraded mode (log +
    skip)" requirement at the level of an individual metric source, so a
    caller building a full sample can simply skip the failed field.
    """
    try:
        return fn()
    except (OSError, psutil.Error, NotImplementedError) as exc:
        logger.warning(
            "Metric source unavailable, continuing in degraded mode",
            extra={"metric_source": source_name, "error": str(exc)},
        )
        return None


def collect_system_metrics() -> SystemMetricSample:
    """Collect one aggregate system-metrics sample.

    Never raises: any individual source that fails is left as ``None`` on
    the resulting sample. The only mandatory fields are
    ``timestamp``, ``cpu_percent``, and ``mem_percent`` — if even the
    lightweight ``cpu_percent()``/``virtual_memory()`` calls fail, that
    indicates a fundamentally broken environment, and the exception is
    allowed to propagate so the Monitoring Engine's outer loop can log a
    cycle-level failure rather than silently emitting a garbage sample.
    """
    timestamp = time.time()

    # Mandatory sources — not wrapped in _safe_source. A failure here is
    # exceptional enough that the caller (the polling loop) should see it.
    cpu_percent = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()

    swap = _safe_source("swap_memory", psutil.swap_memory)
    disk_io = _safe_source("disk_io_counters", psutil.disk_io_counters)
    net_io = _safe_source("net_io_counters", psutil.net_io_counters)
    load_avg = _safe_source("getloadavg", psutil.getloadavg)
    battery = _safe_source("sensors_battery", psutil.sensors_battery)
    cpu_count = _safe_source("cpu_count", psutil.cpu_count)

    return SystemMetricSample(
        timestamp=timestamp,
        cpu_percent=cpu_percent,
        cpu_count_logical=cpu_count,
        mem_total=mem.total,
        mem_used=mem.used,
        mem_available=mem.available,
        mem_percent=mem.percent,
        swap_total=swap.total if swap else None,
        swap_used=swap.used if swap else None,
        swap_percent=swap.percent if swap else None,
        disk_read_bytes=disk_io.read_bytes if disk_io else None,
        disk_write_bytes=disk_io.write_bytes if disk_io else None,
        disk_read_count=disk_io.read_count if disk_io else None,
        disk_write_count=disk_io.write_count if disk_io else None,
        net_bytes_sent=net_io.bytes_sent if net_io else None,
        net_bytes_recv=net_io.bytes_recv if net_io else None,
        net_packets_sent=net_io.packets_sent if net_io else None,
        net_packets_recv=net_io.packets_recv if net_io else None,
        load_avg_1=load_avg[0] if load_avg else None,
        load_avg_5=load_avg[1] if load_avg else None,
        load_avg_15=load_avg[2] if load_avg else None,
        # Aggregate thread count is deliberately left unset here: no
        # lightweight system-wide primitive provides it without a full
        # process enumeration, which runs on a coarser interval
        #. The "thread counts" requirement is satisfied
        # via per-process num_threads in raw_process_metrics instead; this
        # column remains available if a cheap system-wide source is added
        # later.
        thread_count_total=None,
        battery_percent=battery.percent if battery else None,
        power_plugged=battery.power_plugged if battery else None,
    )
