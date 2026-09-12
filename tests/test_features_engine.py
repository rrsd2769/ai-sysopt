from __future__ import annotations

import time

import math
import pandas as pd
import pytest

from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.features.engine import (
    compute_feature_matrix,
    compute_latest_feature_vector,
    feature_matrix_to_records,
)
from aiperfopt.storage.records import SystemMetricSample


def _regular_samples(count: int, interval: float = 2.0, start: float | None = None, **overrides):
    base = start if start is not None else time.time() - count * interval
    samples = []
    for i in range(count):
        ts = base + i * interval
        defaults = dict(
            timestamp=ts,
            cpu_percent=20.0 + (i % 10) * 5.0,
            mem_percent=50.0,
            swap_percent=0.0,
            load_avg_1=1.0,
            disk_read_bytes=1000 * i,
            disk_write_bytes=500 * i,
            net_bytes_sent=2000 * i,
            net_bytes_recv=1500 * i,
        )
        defaults.update(overrides)
        samples.append(SystemMetricSample(**defaults))
    return samples


def test_empty_input_returns_empty_matrix():
    config = FeatureEngineeringConfig()
    matrix = compute_feature_matrix([], config)
    assert matrix.empty


def test_matrix_has_one_row_per_input_sample():
    config = FeatureEngineeringConfig()
    samples = _regular_samples(50)
    matrix = compute_feature_matrix(samples, config)
    assert len(matrix) == 50


def test_gauge_lag_feature_matches_actual_value_n_seconds_ago():
    config = FeatureEngineeringConfig(lookback_windows_seconds=(30, 120), tolerance_seconds=2.0)
    # cpu_percent increases by 1.0 every 2s sample -> value at t-30s should
    # be exactly 15 samples back (30s / 2s per sample).
    base = time.time() - 400
    samples = [
        SystemMetricSample(timestamp=base + i * 2.0, cpu_percent=float(i), mem_percent=50.0)
        for i in range(150)
    ]
    matrix = compute_feature_matrix(samples, config)
    last_row = matrix.iloc[-1]
    # Last sample's cpu_percent == 149; 30s ago (15 samples back) == 134.
    assert last_row["cpu_percent_lag_30s"] == pytest.approx(134.0)


def test_rolling_mean_matches_manual_calculation():
    config = FeatureEngineeringConfig(lookback_windows_seconds=(30,), tolerance_seconds=2.0)
    base = time.time() - 100
    # Constant value -> rolling mean must equal that constant everywhere.
    samples = [
        SystemMetricSample(timestamp=base + i * 2.0, cpu_percent=42.0, mem_percent=50.0)
        for i in range(50)
    ]
    matrix = compute_feature_matrix(samples, config)
    assert (matrix["cpu_percent_roll_mean_30s"] == 42.0).all()


def test_counter_metric_rate_is_computed_not_raw_value():
    config = FeatureEngineeringConfig(lookback_windows_seconds=(30,), tolerance_seconds=2.0)
    base = time.time() - 100
    # disk_read_bytes grows by exactly 1000 bytes every 2s -> 500 bytes/sec.
    samples = [
        SystemMetricSample(
            timestamp=base + i * 2.0, cpu_percent=0.0, mem_percent=0.0, disk_read_bytes=1000 * i
        )
        for i in range(50)
    ]
    matrix = compute_feature_matrix(samples, config)
    # Rate should stabilize at 500 bytes/sec once enough history exists.
    last_row = matrix.iloc[-1]
    assert last_row["disk_read_bytes_rate_roll_mean_30s"] == pytest.approx(500.0)


def test_counter_reset_produces_nan_rate_not_negative_value():
    config = FeatureEngineeringConfig(lookback_windows_seconds=(30,), tolerance_seconds=2.0)
    base = time.time() - 20
    samples = [
        SystemMetricSample(timestamp=base, cpu_percent=0.0, mem_percent=0.0, disk_read_bytes=100_000),
        SystemMetricSample(timestamp=base + 2.0, cpu_percent=0.0, mem_percent=0.0, disk_read_bytes=500),
    ]
    matrix = compute_feature_matrix(samples, config)
    # The rolling stats are built on the rate series; at the reset point
    # the instantaneous rate is NaN, which rolling(min_periods=1) then
    # simply skips (rolling max over one NaN input stays NaN too).
    second_row = matrix.iloc[1]
    assert math.isnan(second_row["disk_read_bytes_rate_roll_max_30s"])


def test_lag_beyond_tolerance_after_gap_is_none():
    """FR-4.3: a lag lookup with no sample nearby must be None, not a stale value."""
    config = FeatureEngineeringConfig(lookback_windows_seconds=(30, 120), tolerance_seconds=3.0)
    base = time.time() - 300
    samples = []
    for i in range(150):
        if 60 <= i < 90:  # ~60s gap
            continue
        samples.append(
            SystemMetricSample(timestamp=base + i * 2.0, cpu_percent=50.0, mem_percent=50.0)
        )
    matrix = compute_feature_matrix(samples, config)

    first_row_after_gap = matrix[matrix["timestamp"] >= base + 90 * 2.0].iloc[0]
    assert math.isnan(first_row_after_gap["cpu_percent_lag_30s"])


def test_rolling_stats_survive_gaps_without_crashing():
    config = FeatureEngineeringConfig(lookback_windows_seconds=(30,), tolerance_seconds=3.0)
    base = time.time() - 300
    samples = []
    for i in range(150):
        if 60 <= i < 90:
            continue
        samples.append(
            SystemMetricSample(timestamp=base + i * 2.0, cpu_percent=50.0, mem_percent=50.0)
        )
    matrix = compute_feature_matrix(samples, config)  # must not raise
    assert not matrix.empty


def test_duplicate_timestamps_are_deduplicated():
    config = FeatureEngineeringConfig()
    base = time.time() - 10
    samples = [
        SystemMetricSample(timestamp=base, cpu_percent=10.0, mem_percent=50.0),
        SystemMetricSample(timestamp=base, cpu_percent=99.0, mem_percent=50.0),  # duplicate ts
        SystemMetricSample(timestamp=base + 2.0, cpu_percent=20.0, mem_percent=50.0),
    ]
    matrix = compute_feature_matrix(samples, config)
    assert len(matrix) == 2  # duplicate collapsed to one row


def test_compute_latest_feature_vector_returns_none_for_empty_input():
    config = FeatureEngineeringConfig()
    assert compute_latest_feature_vector([], config) is None


def test_compute_latest_feature_vector_matches_last_row_of_batch_matrix():
    """FR-4.2: identical live/train computation, not just similar."""
    config = FeatureEngineeringConfig()
    samples = _regular_samples(150, interval=2.0)

    full_matrix = compute_feature_matrix(samples, config)
    last_row = full_matrix.iloc[-1]

    live_window = [
        s for s in samples if s.timestamp >= samples[-1].timestamp - config.min_history_seconds
    ]
    live_vector = compute_latest_feature_vector(live_window, config)

    assert live_vector is not None
    assert live_vector.timestamp == pytest.approx(last_row["timestamp"])
    for key, value in live_vector.feature_values.items():
        full_value = last_row[key]
        full_value = None if (isinstance(full_value, float) and math.isnan(full_value)) else float(full_value)
        if value is None or full_value is None:
            assert value == full_value, f"mismatch on {key}"
        else:
            assert value == pytest.approx(full_value), f"mismatch on {key}"


def test_process_summary_columns_get_the_same_gauge_treatment():
    """Phase 4: an aligned process_summary frame's columns must be
    treated exactly like any other configured gauge metric — lag/
    rolling/rate features come for free, no separate formula path."""
    config = FeatureEngineeringConfig(lookback_windows_seconds=(30,), tolerance_seconds=2.0)
    base = time.time() - 100
    samples = [
        SystemMetricSample(timestamp=base + i * 2.0, cpu_percent=0.0, mem_percent=0.0)
        for i in range(50)
    ]
    process_summary = pd.DataFrame(
        {"timestamp": [s.timestamp for s in samples], "proc_cpu_sum": [42.0] * 50}
    )

    matrix = compute_feature_matrix(samples, config, process_summary=process_summary)

    assert "proc_cpu_sum_roll_mean_30s" in matrix.columns
    assert (matrix["proc_cpu_sum_roll_mean_30s"] == 42.0).all()


def test_process_summary_none_leaves_process_columns_absent():
    """No process data available (e.g. a synthetic training scenario) must
    not fabricate process feature columns."""
    config = FeatureEngineeringConfig()
    samples = _regular_samples(20)
    matrix = compute_feature_matrix(samples, config, process_summary=None)
    assert not any(c.startswith("proc_cpu_sum") for c in matrix.columns)


def test_process_summary_columns_are_causal_like_any_other_gauge():
    """A future process value must not change a past row's engineered
    feature -- the same causality guarantee _feature_bundle already gives
    every gauge, now verified for a merged-in process column too."""
    config = FeatureEngineeringConfig(lookback_windows_seconds=(30,), tolerance_seconds=2.0)
    base = time.time() - 100
    samples = [
        SystemMetricSample(timestamp=base + i * 2.0, cpu_percent=0.0, mem_percent=0.0)
        for i in range(30)
    ]
    timestamps = [s.timestamp for s in samples]

    summary_before = pd.DataFrame({"timestamp": timestamps, "proc_cpu_sum": [10.0] * 30})
    summary_after = summary_before.copy()
    summary_after.loc[summary_after.index[-1], "proc_cpu_sum"] = 9999.0  # only the last row changes

    matrix_before = compute_feature_matrix(samples, config, process_summary=summary_before)
    matrix_after = compute_feature_matrix(samples, config, process_summary=summary_after)

    # Every row except the last (whose own current value legitimately
    # differs) must be unaffected by that future change.
    pd.testing.assert_frame_equal(matrix_before.iloc[:-1], matrix_after.iloc[:-1])


def test_feature_matrix_to_records_round_trips_shape():
    config = FeatureEngineeringConfig()
    samples = _regular_samples(20)
    matrix = compute_feature_matrix(samples, config)
    records = feature_matrix_to_records(matrix, computed_at=1234.0)

    assert len(records) == 20
    assert all(r.computed_at == 1234.0 for r in records)
    assert all(r.feature_set_version == "v2" for r in records)
    assert all(isinstance(r.feature_values, dict) for r in records)


def test_feature_matrix_to_records_empty_matrix_returns_empty_list():
    config = FeatureEngineeringConfig()
    matrix = compute_feature_matrix([], config)
    assert feature_matrix_to_records(matrix) == []


def test_all_nan_metric_column_is_handled_without_erroring():
    """A gauge/counter metric with no real data (e.g. no battery on this
    machine, so battery_percent is None on every sample) must not crash
    feature computation — it should produce an all-NaN feature column."""
    config = FeatureEngineeringConfig(gauge_metrics=("cpu_percent", "battery_percent"))
    samples = _regular_samples(20)  # battery_percent is None on every sample by default
    matrix = compute_feature_matrix(samples, config)  # must not raise
    assert not matrix.empty
    assert matrix["battery_percent_roll_mean_30s"].isna().all()
