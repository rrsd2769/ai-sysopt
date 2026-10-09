"""Small deterministic sample builders for tests.

Samples look like recorded data: regular timestamps, percent gauges, and
cumulative disk/network counters, as psutil reports them.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.storage.records import SystemMetricSample

DEFAULT_START = 1_700_000_000.0

Shape = Callable[[np.ndarray], np.ndarray]


def _default_cpu(i: np.ndarray) -> np.ndarray:
    return 30.0 + 15.0 * np.sin(i / 40.0)


def _default_mem(i: np.ndarray) -> np.ndarray:
    return 50.0 + 5.0 * np.sin(i / 90.0)


def make_samples(
    n: int,
    *,
    start: float = DEFAULT_START,
    interval: float = 2.0,
    cpu: Shape | None = None,
    mem: Shape | None = None,
    seed: int = 0,
) -> list[SystemMetricSample]:
    """``n`` samples ``interval`` seconds apart.

    ``cpu`` / ``mem`` map the sample index array to a percent series (noise
    is added on top); the defaults are a slow sine.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=float)
    cpu_values = np.clip((cpu or _default_cpu)(idx) + rng.normal(0, 1.5, n), 0.0, 100.0)
    mem_values = np.clip((mem or _default_mem)(idx) + rng.normal(0, 0.8, n), 0.0, 100.0)
    disk_rate = np.abs(rng.normal(3_000, 700, n))
    net_rate = np.abs(rng.normal(1_500, 400, n))
    disk_read = np.cumsum(disk_rate * interval)
    disk_write = np.cumsum(disk_rate * 0.6 * interval)
    net_sent = np.cumsum(net_rate * interval)
    net_recv = np.cumsum(net_rate * 0.8 * interval)

    return [
        SystemMetricSample(
            timestamp=start + i * interval,
            cpu_percent=float(cpu_values[i]),
            mem_percent=float(mem_values[i]),
            swap_percent=0.0,
            load_avg_1=float(cpu_values[i] / 100.0 * 4.0),
            disk_read_bytes=int(disk_read[i]),
            disk_write_bytes=int(disk_write[i]),
            net_bytes_sent=int(net_sent[i]),
            net_bytes_recv=int(net_recv[i]),
        )
        for i in range(n)
    ]


def _spiky(base: float, spike: float, period: int, width: int) -> Shape:
    """Repeating pattern: a ramp up to ``spike``, a plateau of ``width``
    samples, then a drop back to ``base``. The ramp gives forecasters a
    trend to learn, so they can beat a persistence baseline."""
    ramp = width

    def shape(i: np.ndarray) -> np.ndarray:
        phase = i % period
        rising = base + (spike - base) * np.clip((phase - (period - ramp - width)) / ramp, 0.0, 1.0)
        return np.where(phase < period - ramp - width, base + 3.0 * np.sin(i / 25.0), rising)

    return shape


def make_training_data(
    n_train: int = 300,
    n_holdout: int = 100,
    *,
    with_spikes: bool = True,
    seed: int = 0,
    interval: float = 2.0,
) -> tuple[dict[str, list[SystemMetricSample]], list[SystemMetricSample]]:
    """One training scenario plus a chronologically later holdout.

    With ``with_spikes``, CPU and memory periodically plateau above the
    contention labelling thresholds, so labels have positive examples.
    """
    thresholds = ContentionLabelingConfig().value_thresholds
    cpu_spike = min(98.0, thresholds["cpu_percent"] + 15.0)
    mem_spike = min(98.0, thresholds["mem_percent"] + 8.0)
    cpu: Shape | None = _spiky(20.0, cpu_spike, period=60, width=15) if with_spikes else None
    mem: Shape | None = _spiky(45.0, mem_spike, period=90, width=12) if with_spikes else None

    train = make_samples(n_train, interval=interval, cpu=cpu, mem=mem, seed=seed)
    holdout_start = DEFAULT_START + n_train * interval
    holdout = make_samples(
        n_holdout, start=holdout_start, interval=interval, cpu=cpu, mem=mem, seed=seed + 1
    )
    return {"train": train}, holdout
