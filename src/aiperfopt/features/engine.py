"""
Feature Engineering computation (Section 7.1, item 4).

Single source of truth
------------------------
:func:`compute_feature_matrix` is the *only* place feature formulas are
defined. Both consumers use it identically (FR-4.2):

- **Live inference** (:func:`compute_latest_feature_vector`): call it on a
  short trailing window read from storage, take the last row.
- **Batch/training**: call it directly on a large historical range; every
  row is a training example.

There is no second, hand-optimized "streaming" implementation of these
formulas to drift out of sync with the batch one — the live path is
simply the batch path applied to a smaller input.

Gauges vs. counters (FR-4.1)
------------------------------
``cpu_percent``, ``mem_percent``, ``swap_percent``, and ``load_avg_1`` are
instantaneous gauges: lag/rolling statistics are computed directly on
the raw value.

``disk_read_bytes``, ``disk_write_bytes``, ``net_bytes_sent``, and
``net_bytes_recv`` are monotonically-increasing OS counters (as returned
by ``psutil.disk_io_counters()`` / ``net_io_counters()``, cumulative
since boot). A rolling mean of a raw cumulative counter is meaningless —
it is first differenced into a per-second rate
(:func:`counter_rate_per_second`), and lag/rolling statistics are
computed on *that* derived rate series instead. A negative delta (the
counter went down — e.g. after a reboot resets it) is treated as a gap,
not a real negative rate.

Gap handling (FR-4.3)
------------------------
Every window in this module is time-based (seconds), not row-count-based,
because the polling interval is configurable and individual cycles can
be skipped (FR-1.4 degradation, dropped samples under backpressure).
Two gap-handling rules follow from that:

- A lag lookup ("value ~30s ago") uses ``pandas.merge_asof`` with an
  explicit time tolerance. If no sample exists within that tolerance of
  the target time, the feature is ``None`` — never a stale, far-away
  sample silently presented as if it were recent (this is precisely what
  FR-4.3 prohibits).
- A counter's per-second rate is only computed between two samples less
  than ``max_gap_seconds`` apart; across a larger gap the "rate" would
  silently average over an unknown amount of missed activity.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Sequence

import numpy as np
import pandas as pd

from aiperfopt.common.timeutils import epoch_seconds_to_utc_datetime
from aiperfopt.features.config import FEATURE_SET_VERSION, FeatureEngineeringConfig
from aiperfopt.storage.records import FeatureVectorRecord, SystemMetricSample


def _samples_to_frame(samples: Sequence[SystemMetricSample]) -> pd.DataFrame:
    """Build a chronologically-sorted, timestamp-deduplicated, time-indexed frame."""
    if not samples:
        return pd.DataFrame()

    df = pd.DataFrame.from_records([asdict(s) for s in samples])
    # Later samples win on a duplicate timestamp (shouldn't normally
    # happen given the polling interval, but is possible under clock
    # skew / test scenarios — silently ignoring the duplicate rather than
    # erroring keeps this function robust for FR-4.3's spirit).
    df = df.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
    df["_dt"] = epoch_seconds_to_utc_datetime(df["timestamp"])
    return df.set_index("_dt")


def _lag_with_tolerance(series: pd.Series, lag_seconds: int, tolerance_seconds: float) -> pd.Series:
    """Value approximately ``lag_seconds`` before each index point, or NaN if none nearby."""
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)

    targets = pd.DataFrame({"target_time": series.index - pd.Timedelta(seconds=lag_seconds)})
    source = valid.rename("value").reset_index().rename(columns={"_dt": "sample_time"})

    merged = pd.merge_asof(
        targets.sort_values("target_time"),
        source.sort_values("sample_time"),
        left_on="target_time",
        right_on="sample_time",
        direction="backward",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    )
    # merge_asof sorted `targets`; realign back to the original row order.
    merged.index = targets.sort_values("target_time").index
    return merged["value"].reindex(range(len(series))).set_axis(series.index)


def counter_rate_per_second(series: pd.Series, max_gap_seconds: float) -> pd.Series:
    """Per-second delta of a cumulative counter; NaN across resets or large gaps."""
    dt_seconds = series.index.to_series().diff().dt.total_seconds()
    dv = series.diff()
    rate = dv / dt_seconds

    invalid = (dv < 0) | (dt_seconds > max_gap_seconds) | dt_seconds.isna()
    rate = rate.mask(invalid)
    return rate


def _feature_bundle(
    series: pd.Series, prefix: str, windows: tuple[int, ...], tolerance_seconds: float
) -> dict[str, pd.Series]:
    """Standard lag + rolling-stat + rate-of-change bundle applied to one metric series.

    Applied identically whether ``series`` is a raw gauge or an already
    rate-transformed counter — this is what keeps gauge and counter
    handling consistent (FR-4.1) despite counters needing a
    differencing step first.
    """
    out: dict[str, pd.Series] = {}
    lags: dict[int, pd.Series] = {}

    for window in windows:
        lag = _lag_with_tolerance(series, window, tolerance_seconds)
        lags[window] = lag
        out[f"{prefix}_lag_{window}s"] = lag

        window_label = f"{window}s"
        out[f"{prefix}_roll_mean_{window_label}"] = series.rolling(window_label, min_periods=1).mean()
        out[f"{prefix}_roll_std_{window_label}"] = series.rolling(window_label, min_periods=2).std()
        out[f"{prefix}_roll_min_{window_label}"] = series.rolling(window_label, min_periods=1).min()
        out[f"{prefix}_roll_max_{window_label}"] = series.rolling(window_label, min_periods=1).max()

    # Rate-of-change against the shortest configured window: (now - lag) / window.
    # The window's nominal length is used as the time base rather than the
    # lagged sample's *actual* offset (which may differ by up to
    # tolerance_seconds) — a documented simplification that keeps this
    # formula simple; the tolerance is small relative to the window by
    # default (5s against a 30s window).
    shortest = min(windows)
    out[f"{prefix}_rate_of_change"] = (series - lags[shortest]) / shortest

    return out


def compute_feature_matrix(
    samples: Sequence[SystemMetricSample],
    config: FeatureEngineeringConfig,
    process_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute one feature row per input raw sample. The single source of truth (FR-4.2).

    ``process_summary`` (Phase 4), if given, is a process-summary frame
    already aligned one-to-one onto these samples' own timestamps (see
    :func:`aiperfopt.features.process_features.align_process_summary_to_timestamps`)
    — its columns are joined onto the raw sample frame *before* the
    gauge/counter loop below runs, so any of its column names already
    present in ``config.gauge_metrics`` get the same lag/rolling/rate
    treatment as any other gauge, with no separate formula path. Passing
    ``None`` (the default) computes system-metrics-only features exactly
    as before Phase 4 — e.g. for synthetic scenarios, which have no
    process data to align.

    Returns a DataFrame with a ``timestamp`` column plus one column per
    engineered feature. Empty input returns an empty DataFrame with no
    columns — callers should check ``.empty`` before use.
    """
    df = _samples_to_frame(samples)
    if df.empty:
        return pd.DataFrame()

    if process_summary is not None and not process_summary.empty:
        df = df.join(process_summary.set_index("timestamp"), on="timestamp")

    feature_columns: dict[str, pd.Series] = {}

    for metric in config.gauge_metrics:
        if metric not in df.columns:
            continue
        series = df[metric].astype(float)
        feature_columns.update(
            _feature_bundle(series, metric, config.lookback_windows_seconds, config.tolerance_seconds)
        )

    for metric in config.counter_metrics:
        if metric not in df.columns:
            continue
        raw_counter = df[metric].astype(float)
        rate = counter_rate_per_second(raw_counter, config.max_gap_seconds)
        feature_columns.update(
            _feature_bundle(
                rate, f"{metric}_rate", config.lookback_windows_seconds, config.tolerance_seconds
            )
        )

    features_df = pd.DataFrame(feature_columns, index=df.index)
    features_df.insert(0, "timestamp", df["timestamp"].to_numpy())
    return features_df.reset_index(drop=True)


def _row_to_feature_values(row: pd.Series) -> dict[str, float]:
    """Convert a feature-matrix row to a JSON-serializable dict, NaN -> None."""
    values: dict[str, float] = {}
    for column, value in row.items():
        if column == "timestamp":
            continue
        values[column] = None if pd.isna(value) else float(value)
    return values


def compute_latest_feature_vector(
    samples: Sequence[SystemMetricSample],
    config: FeatureEngineeringConfig,
    process_summary: pd.DataFrame | None = None,
) -> FeatureVectorRecord | None:
    """Feature vector for the most recent sample in ``samples`` — the live-inference call.

    Returns ``None`` if ``samples`` is empty. Callers are responsible for
    supplying enough trailing history (see
    :attr:`FeatureEngineeringConfig.min_history_seconds`) — this function
    does not fetch data itself, keeping it testable without storage (see
    :func:`compute_live_feature_vector` for the storage-backed
    convenience wrapper). ``process_summary``: see
    :func:`compute_feature_matrix`.
    """
    matrix = compute_feature_matrix(samples, config, process_summary=process_summary)
    if matrix.empty:
        return None

    last_row = matrix.iloc[-1]
    return FeatureVectorRecord(
        timestamp=float(last_row["timestamp"]),
        computed_at=time.time(),
        feature_set_version=FEATURE_SET_VERSION,
        feature_values=_row_to_feature_values(last_row),
    )


def feature_matrix_to_records(
    matrix: pd.DataFrame, computed_at: float | None = None
) -> list[FeatureVectorRecord]:
    """Convert every row of a batch-computed feature matrix into storable records.

    Used for backfilling the ``features`` table over historical data
    (e.g. ahead of Forecasting model training in Phase 4).
    """
    if matrix.empty:
        return []
    ts = computed_at if computed_at is not None else time.time()
    return [
        FeatureVectorRecord(
            timestamp=float(row["timestamp"]),
            computed_at=ts,
            feature_set_version=FEATURE_SET_VERSION,
            feature_values=_row_to_feature_values(row),
        )
        for _, row in matrix.iterrows()
    ]
