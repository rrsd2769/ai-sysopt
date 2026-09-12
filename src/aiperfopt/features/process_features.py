"""
Process summary series — pure functions turning a window of raw
process-level data into a compact, per-poll-timestamp summary (Phase 2 of
``docs/plans/2026-09-12-presentable-predictive-monitor.md``), plus the
Phase 4 alignment step that re-expresses that summary on an arbitrary
timeline (the system-metric one, in practice).

Two timelines, two functions
------------------------------
Process metrics are polled on their own loop
(``AppSettings.process_polling_interval_seconds``), independent of the
system-metrics loop (see ``aiperfopt.monitoring.engine``) — the two sets
of timestamps generally don't coincide. :func:`compute_process_summary_series`
produces one row per *process-poll* timestamp (Phase 2's durability
concern, below). :func:`align_process_summary_to_timestamps` re-samples
that per-poll series onto a *different*, caller-supplied set of
timestamps — what Feature Engineering (Phase 4) needs to feed these
columns through :func:`aiperfopt.features.engine.compute_feature_matrix`
alongside the system metrics, which are keyed by the system-metric
timeline. Keeping these as two functions (rather than one that takes an
optional "eval timestamps" override) keeps Phase 2's persistence path
byte-for-byte unchanged: it always aligns to its own poll timestamps at
zero tolerance, exactly as it always has.

This module's job is narrower and more urgent: ``raw_process_metrics``
and ``process_events`` are pruned after a short retention window (see
``aiperfopt.storage.rollup``'s process retention, independent of system
metrics' much longer one) so that multi-day passive collection stays
storage-bounded. Multi-day training data is the entire point of the
pivot this plan describes, so the *derived* process-level signal has to
be computed and persisted durably before the raw rows it comes from are
deleted — see :func:`aiperfopt.features.storage_bridge.
summarize_and_persist_process_features` for where that happens.

Working at the process poll's own timestamp, not the system-metric
timeline, keeps this module a pure function of process-level tables only
(NFR-14.1's testability-in-isolation, same rationale as
``aiperfopt.features.engine`` for system metrics) — it has no dependency
on ``FeatureEngineeringConfig`` and doesn't need to know the system
polling interval to do its job correctly.

Causality
---------
Every column is computed from samples with timestamp <= the row's own
timestamp (event-derived columns use strictly backward-looking windows:
"launches in the 30s *before* t"). No feature at time t is ever affected
by a sample recorded after t.

Reconstructing whole-population sums under the Phase 2 storage bound
----------------------------------------------------------------------
``process_metrics`` only contains the top-N-by-CPU union top-N-by-memory
rows that :func:`aiperfopt.monitoring.process_tracker.bound_process_metrics`
decided to keep (Phase 2's storage bound) — not every process. Passing
the accompanying ``process_aggregates`` ("everything else") rows lets
whole-population sums/counts (``proc_cpu_sum``, ``proc_mem_rss_sum``,
``proc_count_active``) be reconstructed exactly by adding the aggregate
back in; top-K columns (``proc_cpu_top1``, ...) are exact without it
since the true top-K by CPU/memory is always among the retained rows by
construction. ``proc_threads_sum`` is the one column that is *not*
population-exact under truncation — the aggregate row doesn't track a
thread-count sum for the "other" bucket (see ``ProcessMetricAggregate``)
— so it reflects only the retained top-N processes' threads, a
documented, deliberate approximation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

PROCESS_SUMMARY_COLUMNS: tuple[str, ...] = (
    "proc_launch_count_30s",
    "proc_launch_count_120s",
    "proc_terminate_count_30s",
    "proc_launch_burst_10s",
    "proc_young_count",
    "proc_young_cpu_sum",
    "proc_cpu_sum",
    "proc_cpu_top1",
    "proc_cpu_top1_share",
    "proc_cpu_top5_share",
    "proc_count_above_20pct",
    "proc_count_active",
    "proc_mem_rss_sum",
    "proc_mem_top1_rss",
    "proc_threads_sum",
)

_YOUNG_WINDOW_SECONDS = 60.0
_ABOVE_PCT_THRESHOLD = 20.0
_LAUNCH_COUNT_WINDOWS = {"proc_launch_count_30s": 30.0, "proc_launch_count_120s": 120.0}
_TERMINATE_COUNT_WINDOWS = {"proc_terminate_count_30s": 30.0}
_BURST_LOOKBACK_SECONDS = 30.0
_BURST_WINDOW_SECONDS = 10.0

_PROCESS_METRICS_COLUMNS = ("timestamp", "pid", "cpu_percent", "memory_rss", "num_threads", "create_time")
_PROCESS_EVENTS_COLUMNS = ("timestamp", "event_type")
_PROCESS_AGGREGATE_COLUMNS = ("timestamp", "other_count", "other_cpu_sum", "other_mem_rss_sum")


def _empty_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", *PROCESS_SUMMARY_COLUMNS])


def _max_count_in_any_subwindow(timestamps: list[float], subwindow_seconds: float) -> int:
    """Max number of ``timestamps`` falling within any window of width ``subwindow_seconds``.

    Standard sorted two-pointer sliding window — O(n log n) for the sort,
    O(n) thereafter.
    """
    if not timestamps:
        return 0
    ordered = sorted(timestamps)
    best = 0
    left = 0
    for right, t in enumerate(ordered):
        while t - ordered[left] > subwindow_seconds:
            left += 1
        best = max(best, right - left + 1)
    return best


def _agg_by_ts(process_aggregates: pd.DataFrame) -> dict[float, dict[str, float]]:
    agg_by_ts: dict[float, dict[str, float]] = {}
    if not process_aggregates.empty:
        for _, row in process_aggregates.iterrows():
            agg_by_ts[float(row["timestamp"])] = {
                "other_count": float(row.get("other_count", 0) or 0),
                "other_cpu_sum": float(row.get("other_cpu_sum", 0.0) or 0.0),
                "other_mem_rss_sum": float(row.get("other_mem_rss_sum", 0) or 0),
            }
    return agg_by_ts


_ZERO_AGG = {"other_count": 0.0, "other_cpu_sum": 0.0, "other_mem_rss_sum": 0.0}


def _launches_and_terminates(process_events: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    launches = pd.Series(dtype="float64")
    terminates = pd.Series(dtype="float64")
    if not process_events.empty:
        launches = process_events.loc[process_events["event_type"] == "launch", "timestamp"]
        terminates = process_events.loc[process_events["event_type"] == "terminate", "timestamp"]
    return launches, terminates


def _summary_row_at(
    poll_ts: float, at_t: pd.DataFrame, other: dict[str, float], launches: pd.Series, terminates: pd.Series
) -> dict[str, float]:
    """One summary row for a poll timestamp ``poll_ts`` whose own process rows are ``at_t``.

    Shared by :func:`compute_process_summary_series` (evaluated at every
    poll's own timestamp) and :func:`align_process_summary_to_timestamps`'s
    internal per-poll pass (evaluated once per poll, then re-sampled onto
    the caller's timestamps) — this is the one place every
    :data:`PROCESS_SUMMARY_COLUMNS` formula is defined.
    """

    def _numeric(column: str, fill: float) -> pd.Series:
        if column not in at_t:
            return pd.Series(dtype="float64")
        return pd.to_numeric(at_t[column], errors="coerce").fillna(fill)

    cpu = _numeric("cpu_percent", 0.0)
    cpu_sorted = cpu.sort_values(ascending=False)
    cpu_top1 = float(cpu_sorted.iloc[0]) if not cpu_sorted.empty else 0.0
    cpu_top5_sum = float(cpu_sorted.head(5).sum())
    cpu_sum = float(cpu.sum()) + other["other_cpu_sum"]
    cpu_top1_share = (cpu_top1 / cpu_sum) if cpu_sum > 0 else 0.0
    cpu_top5_share = (cpu_top5_sum / cpu_sum) if cpu_sum > 0 else 0.0

    mem = _numeric("memory_rss", 0.0)
    mem_sum = float(mem.sum()) + other["other_mem_rss_sum"]
    mem_top1 = float(mem.max()) if not mem.empty else 0.0

    threads = _numeric("num_threads", 0.0)

    if "create_time" in at_t:
        age = poll_ts - _numeric("create_time", float("inf"))
        young_mask = (age >= 0) & (age <= _YOUNG_WINDOW_SECONDS)
    else:
        young_mask = pd.Series(False, index=at_t.index)
    young_cpu_sum = float(cpu[young_mask].sum())
    young_count = int(young_mask.sum())

    def _count_in_window(series: pd.Series, window_seconds: float) -> int:
        if series.empty:
            return 0
        return int(series.between(poll_ts - window_seconds, poll_ts, inclusive="right").sum())

    def _launches_in_lookback() -> list[float]:
        if launches.empty:
            return []
        windowed = launches[launches.between(poll_ts - _BURST_LOOKBACK_SECONDS, poll_ts, inclusive="right")]
        return windowed.tolist()

    return {
        "timestamp": poll_ts,
        "proc_launch_count_30s": _count_in_window(launches, _LAUNCH_COUNT_WINDOWS["proc_launch_count_30s"]),
        "proc_launch_count_120s": _count_in_window(launches, _LAUNCH_COUNT_WINDOWS["proc_launch_count_120s"]),
        "proc_terminate_count_30s": _count_in_window(
            terminates, _TERMINATE_COUNT_WINDOWS["proc_terminate_count_30s"]
        ),
        "proc_launch_burst_10s": _max_count_in_any_subwindow(_launches_in_lookback(), _BURST_WINDOW_SECONDS),
        "proc_young_count": young_count,
        "proc_young_cpu_sum": young_cpu_sum,
        "proc_cpu_sum": cpu_sum,
        "proc_cpu_top1": cpu_top1,
        "proc_cpu_top1_share": cpu_top1_share,
        "proc_cpu_top5_share": cpu_top5_share,
        "proc_count_above_20pct": int((cpu >= _ABOVE_PCT_THRESHOLD).sum()),
        "proc_count_active": int(len(at_t)) + int(other["other_count"]),
        "proc_mem_rss_sum": mem_sum,
        "proc_mem_top1_rss": mem_top1,
        "proc_threads_sum": float(threads.sum()),
    }


def compute_process_summary_series(
    process_metrics: pd.DataFrame,
    process_events: pd.DataFrame,
    process_aggregates: pd.DataFrame,
) -> pd.DataFrame:
    """One summary row per distinct timestamp present in ``process_metrics``.

    Parameters
    ----------
    process_metrics:
        Rows shaped like ``raw_process_metrics`` — must have at least
        ``timestamp``, ``pid``, ``cpu_percent``; ``memory_rss``,
        ``num_threads``, ``create_time`` are treated as 0/absent if
        missing or null. May be top-N-bounded (Phase 2) rather than a
        full per-process enumeration.
    process_events:
        Rows shaped like ``process_events`` (``timestamp``, ``event_type``
        of "launch"/"terminate") — always a complete enumeration,
        unaffected by the Phase 2 storage bound. May be empty.
    process_aggregates:
        Rows shaped like ``raw_process_metrics_other`` (``timestamp``,
        ``other_count``, ``other_cpu_sum``, ``other_mem_rss_sum``) — at
        most one row per poll timestamp. May be empty (e.g. for data
        collected before Phase 2, or a poll with nothing truncated).
    """
    if process_metrics.empty:
        return _empty_summary_frame()

    poll_timestamps = sorted(process_metrics["timestamp"].unique())
    agg_by_ts = _agg_by_ts(process_aggregates)
    launches, terminates = _launches_and_terminates(process_events)

    rows = [
        _summary_row_at(
            ts,
            process_metrics[process_metrics["timestamp"] == ts],
            agg_by_ts.get(ts, _ZERO_AGG),
            launches,
            terminates,
        )
        for ts in poll_timestamps
    ]
    return pd.DataFrame(rows)


def align_process_summary_to_timestamps(
    summary: pd.DataFrame, eval_timestamps: Sequence[float], tolerance_seconds: float
) -> pd.DataFrame:
    """Re-sample an already-computed per-poll-timestamp ``summary`` onto ``eval_timestamps``.

    For each timestamp in ``eval_timestamps``, uses the summary row from
    the most recent process poll at or before it, provided that poll is
    within ``tolerance_seconds``. This is a backward-looking (causal)
    match only — a later poll never fills an earlier evaluation point —
    and a poll farther away than the tolerance leaves every
    :data:`PROCESS_SUMMARY_COLUMNS` value ``NaN`` rather than
    carrying a stale value forward indefinitely.

    ``summary`` is expected to already be shaped like
    :func:`compute_process_summary_series`'s output (a ``timestamp``
    column plus :data:`PROCESS_SUMMARY_COLUMNS`) — it may come from that
    function directly (still-unpruned raw data) or from reloaded
    :class:`~aiperfopt.storage.records.ProcessFeatureSummaryRecord` rows
    (already-pruned historical data); both share the same column shape.
    """
    eval_ts = pd.Index(pd.Series(eval_timestamps, dtype="float64"), name="timestamp")
    if summary.empty:
        return pd.DataFrame(
            {"timestamp": eval_ts, **{col: np.nan for col in PROCESS_SUMMARY_COLUMNS}}
        )

    left = pd.DataFrame({"timestamp": eval_ts}).sort_values("timestamp")
    right = summary[["timestamp", *PROCESS_SUMMARY_COLUMNS]].sort_values("timestamp")
    merged = pd.merge_asof(
        left, right, on="timestamp", direction="backward", tolerance=float(tolerance_seconds)
    )
    return merged.sort_values("timestamp").reset_index(drop=True)
