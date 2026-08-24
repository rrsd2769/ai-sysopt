"""
Synthetic scenario generation (Section 10.3, "Phase 1 — Synthetic
Bootstrapping").

There is no accumulated real-usage data for a brand-new install (Section
10.3's Phase 2 is explicitly not required for a working V1 demo — see
R-3's mitigation in Section 16). Forecasting, Contention Prediction, and
Anomaly Detection all need *something* to train and evaluate against, so
this module generates controlled, deliberately varied stress-test
scenarios: different workload shapes, intensities, and a benign
non-contention spike, exactly as Section 10.3 describes.

This module originally lived under ``forecasting`` (its first consumer)
and was promoted to this top-level location once Contention Prediction
and Anomaly Detection became a second and third consumer (Phase 5) —
following the project's stated preference for deferring structure until
a second real consumer justifies it, rather than guessing the right
package boundary up front.

Every generator is deterministic given a seed, produces an in-memory list
of :class:`SystemMetricSample` (never written to ``raw_system_metrics`` —
see :mod:`aiperfopt.forecasting.training_pipeline`'s module docstring for
why mixing synthetic and real data in that table would be a real bug),
and returns samples at a fixed, regular interval — a simplifying
assumption real monitoring data does not always satisfy (Phase 2's
degradation/backpressure can produce gaps), but a reasonable one for
generating clean initial training signal.
"""

from __future__ import annotations

import numpy as np

from aiperfopt.storage.records import SystemMetricSample


def _clip_percent(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 100.0)


def _base_series(
    rng: np.random.Generator,
    duration_seconds: float,
    interval_seconds: float,
    start_timestamp: float,
) -> np.ndarray:
    n = int(duration_seconds / interval_seconds)
    return start_timestamp + np.arange(n) * interval_seconds


def generate_idle_scenario(
    duration_seconds: float = 900.0,
    interval_seconds: float = 2.0,
    start_timestamp: float = 0.0,
    seed: int = 1,
) -> list[SystemMetricSample]:
    """Steady, low-utilization baseline — the "nothing is happening" scenario."""
    rng = np.random.default_rng(seed)
    timestamps = _base_series(rng, duration_seconds, interval_seconds, start_timestamp)
    n = len(timestamps)

    cpu = _clip_percent(5.0 + rng.normal(0, 1.5, n))
    mem = _clip_percent(30.0 + rng.normal(0, 1.0, n))
    disk_rate = np.abs(rng.normal(2_000, 500, n))
    net_rate = np.abs(rng.normal(1_000, 300, n))

    return _assemble(timestamps, cpu, mem, disk_rate, disk_rate * 0.6, net_rate, net_rate * 0.8)


def generate_cpu_stress_scenario(
    duration_seconds: float = 900.0,
    interval_seconds: float = 2.0,
    start_timestamp: float = 0.0,
    seed: int = 2,
) -> list[SystemMetricSample]:
    """CPU ramps up partway through and stays high — the flagship "wow moment" pattern.

    This is the scenario Section 6's flagship demo describes: a workload
    that, viewed early enough, shows a clear rising trend a forecaster
    should catch before saturation.
    """
    rng = np.random.default_rng(seed)
    timestamps = _base_series(rng, duration_seconds, interval_seconds, start_timestamp)
    n = len(timestamps)
    ramp_start = n // 3
    ramp_length = n // 4

    cpu = np.full(n, 8.0)
    cpu[ramp_start : ramp_start + ramp_length] = np.linspace(8.0, 85.0, ramp_length)
    cpu[ramp_start + ramp_length :] = 85.0
    cpu = _clip_percent(cpu + rng.normal(0, 2.0, n))

    mem = _clip_percent(35.0 + rng.normal(0, 1.5, n))
    disk_rate = np.abs(rng.normal(3_000, 700, n))
    net_rate = np.abs(rng.normal(1_500, 400, n))

    return _assemble(timestamps, cpu, mem, disk_rate, disk_rate * 0.5, net_rate, net_rate * 0.7)


def generate_memory_pressure_gradual_scenario(
    duration_seconds: float = 1200.0,
    interval_seconds: float = 2.0,
    start_timestamp: float = 0.0,
    seed: int = 3,
) -> list[SystemMetricSample]:
    """Memory trends steadily upward toward saturation over the whole run (Section 12's
    "Memory Pressure — Gradual" diagnosis category's triggering pattern)."""
    rng = np.random.default_rng(seed)
    timestamps = _base_series(rng, duration_seconds, interval_seconds, start_timestamp)
    n = len(timestamps)

    mem = _clip_percent(np.linspace(30.0, 92.0, n) + rng.normal(0, 1.0, n))
    cpu = _clip_percent(15.0 + rng.normal(0, 3.0, n))
    disk_rate = np.abs(rng.normal(4_000, 1_000, n))
    net_rate = np.abs(rng.normal(1_200, 300, n))

    return _assemble(timestamps, cpu, mem, disk_rate, disk_rate * 0.7, net_rate, net_rate * 0.9)


def generate_disk_io_burst_scenario(
    duration_seconds: float = 900.0,
    interval_seconds: float = 2.0,
    start_timestamp: float = 0.0,
    seed: int = 4,
) -> list[SystemMetricSample]:
    """Intermittent heavy disk I/O bursts (Section 12's "Disk I/O Bottleneck" pattern)."""
    rng = np.random.default_rng(seed)
    timestamps = _base_series(rng, duration_seconds, interval_seconds, start_timestamp)
    n = len(timestamps)

    disk_read = np.abs(rng.normal(3_000, 800, n))
    disk_write = np.abs(rng.normal(2_000, 600, n))
    # Three bursts of sustained heavy I/O.
    for burst_start in (n // 5, 2 * n // 5, 3 * n // 4):
        burst_len = n // 12
        disk_read[burst_start : burst_start + burst_len] += rng.uniform(80_000, 150_000, burst_len)
        disk_write[burst_start : burst_start + burst_len] += rng.uniform(40_000, 90_000, burst_len)

    cpu = _clip_percent(20.0 + rng.normal(0, 3.0, n))
    mem = _clip_percent(45.0 + rng.normal(0, 2.0, n))
    net_rate = np.abs(rng.normal(1_000, 300, n))

    return _assemble(timestamps, cpu, mem, disk_read, disk_write, net_rate, net_rate * 0.8)


def generate_benign_spike_scenario(
    duration_seconds: float = 600.0,
    interval_seconds: float = 2.0,
    start_timestamp: float = 0.0,
    seed: int = 5,
) -> list[SystemMetricSample]:
    """A short-lived spike that fully resolves — a non-contention pattern.

    Included per Section 10.3's "benign non-contention spikes" — useful
    training signal for a model that must learn that not every transient
    spike is the start of sustained contention.
    """
    rng = np.random.default_rng(seed)
    timestamps = _base_series(rng, duration_seconds, interval_seconds, start_timestamp)
    n = len(timestamps)
    spike_start = n // 2
    spike_len = max(5, n // 30)

    cpu = np.full(n, 12.0)
    cpu[spike_start : spike_start + spike_len] = 75.0
    cpu = _clip_percent(cpu + rng.normal(0, 2.0, n))

    mem = _clip_percent(30.0 + rng.normal(0, 1.5, n))
    disk_rate = np.abs(rng.normal(2_000, 500, n))
    net_rate = np.abs(rng.normal(1_000, 300, n))

    return _assemble(timestamps, cpu, mem, disk_rate, disk_rate * 0.6, net_rate, net_rate * 0.8)


def _assemble(
    timestamps: np.ndarray,
    cpu: np.ndarray,
    mem: np.ndarray,
    disk_read_rate: np.ndarray,
    disk_write_rate: np.ndarray,
    net_sent_rate: np.ndarray,
    net_recv_rate: np.ndarray,
) -> list[SystemMetricSample]:
    """Integrate per-second rates into monotonically-increasing cumulative
    counters, matching what psutil's real cumulative counters look like —
    Feature Engineering's counter-rate transform (FR-4.1) expects
    cumulative input, not pre-computed rates."""
    interval = float(timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 1.0
    disk_read_cum = np.cumsum(disk_read_rate * interval)
    disk_write_cum = np.cumsum(disk_write_rate * interval)
    net_sent_cum = np.cumsum(net_sent_rate * interval)
    net_recv_cum = np.cumsum(net_recv_rate * interval)

    samples = []
    for i, ts in enumerate(timestamps):
        samples.append(
            SystemMetricSample(
                timestamp=float(ts),
                cpu_percent=float(cpu[i]),
                mem_percent=float(mem[i]),
                swap_percent=0.0,
                load_avg_1=float(cpu[i] / 100.0 * 4.0),
                disk_read_bytes=int(disk_read_cum[i]),
                disk_write_bytes=int(disk_write_cum[i]),
                net_bytes_sent=int(net_sent_cum[i]),
                net_bytes_recv=int(net_recv_cum[i]),
            )
        )
    return samples


# Registry used by the training pipeline. "cpu_stress" is reserved as the
# held-out generalization scenario (Section 11: "at least one full
# synthetic scenario held out entirely from all training folds") — it
# never appears in a walk-forward training fold, only in the final
# generalization check.
SCENARIO_GENERATORS = {
    "idle": generate_idle_scenario,
    "memory_pressure_gradual": generate_memory_pressure_gradual_scenario,
    "disk_io_burst": generate_disk_io_burst_scenario,
    "benign_spike": generate_benign_spike_scenario,
}

HELD_OUT_SCENARIO_NAME = "cpu_stress"
HELD_OUT_SCENARIO_GENERATOR = generate_cpu_stress_scenario
