"""
Contributing processes lookup (FR-6.2's "list of contributing... processes").

Not a learned model — a straightforward lookup against recent
``raw_process_metrics``, resolved once the dominant contributing *metric*
is already known (from :class:`~aiperfopt.contention.model.ContentionClassifier`'s
coefficient-based attribution). This matches the flagship demo's own
framing (Section 6): "Primary contributing processes are X, Y, Z" is
presented as a straightforward attribution, not as a separately-trained
per-process prediction — which the spec never asks for and which would
require an entirely different, much larger-scoped model (per-process
time series across an unbounded, constantly-changing set of processes).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine

from aiperfopt.storage.reader import get_process_metrics_range

# Maps a contention-relevant metric name to the raw_process_metrics column
# that best explains "which processes are driving this metric." Counter
# metrics (disk/net) don't have a direct per-process byte-rate column in
# the current schema (Section 9's process-level metric set is CPU/memory/
# threads/priority — see PROJECT_SPEC.md Section 9), so they fall back to
# CPU as the most broadly available proxy for "which processes are busy
# right now" rather than returning nothing.
_METRIC_TO_PROCESS_COLUMN = {
    "cpu_percent": "cpu_percent",
    "mem_percent": "memory_percent",
    "swap_percent": "memory_percent",
    "load_avg_1": "cpu_percent",
}


@dataclass(frozen=True)
class ContributingProcess:
    pid: int
    name: str
    value: float  # the value of whichever column was ranked on (e.g. cpu_percent)


def get_contributing_processes(
    engine: Engine, as_of: float, dominant_metric: str, top_n: int, lookback_seconds: float = 10.0
) -> list[ContributingProcess]:
    """Top ``top_n`` processes by whichever column best explains ``dominant_metric``,
    from the most recent process snapshot at or before ``as_of``.
    """
    column_name = _METRIC_TO_PROCESS_COLUMN.get(dominant_metric.replace("_rate", ""), "cpu_percent")

    rows = get_process_metrics_range(engine, start=as_of - lookback_seconds, end=as_of)
    if not rows:
        return []

    latest_per_pid: dict[int, object] = {}
    for row in rows:
        existing = latest_per_pid.get(row.pid)
        if existing is None or row.timestamp > existing.timestamp:
            latest_per_pid[row.pid] = row

    def _value(row) -> float:
        return getattr(row, column_name, None) or 0.0

    ranked = sorted(latest_per_pid.values(), key=_value, reverse=True)[:top_n]
    return [ContributingProcess(pid=r.pid, name=r.name, value=_value(r)) for r in ranked]
