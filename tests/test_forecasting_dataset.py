from __future__ import annotations

import pytest

from aiperfopt.features import FeatureEngineeringConfig, compute_feature_matrix
from aiperfopt.forecasting.dataset import build_supervised_dataset
from aiperfopt.storage.records import SystemMetricSample


def _samples(n: int, interval: float = 2.0, start: float = 0.0):
    return [
        SystemMetricSample(timestamp=start + i * interval, cpu_percent=float(i), mem_percent=50.0)
        for i in range(n)
    ]


@pytest.mark.plumbing
def test_whole_second_timestamps_do_not_raise_merge_error():
    """Regression test: pandas 3.0's datetime resolution inference upcasts
    ``index - Timedelta`` to a different resolution than a fresh
    to_datetime() call when every input timestamp happens to be a whole
    number — synthetic data (unlike time.time()-based real timestamps)
    reliably has whole-number timestamps, so this must not raise
    MergeError."""
    samples = _samples(100, interval=2.0, start=0.0)  # deliberately whole-second
    config = FeatureEngineeringConfig(lookback_windows_seconds=(10, 30), tolerance_seconds=3.0)
    fm = compute_feature_matrix(samples, config)

    dataset = build_supervised_dataset(samples, fm, "cpu_percent", horizon_seconds=10, tolerance_seconds=3.0)
    assert not dataset.empty


@pytest.mark.behavioural
def test_target_matches_actual_future_value():
    samples = _samples(100, interval=2.0)  # cpu_percent == sample index
    config = FeatureEngineeringConfig(lookback_windows_seconds=(10, 30), tolerance_seconds=2.0)
    fm = compute_feature_matrix(samples, config)

    dataset = build_supervised_dataset(samples, fm, "cpu_percent", horizon_seconds=10, tolerance_seconds=2.0)
    # 10s ahead at 2s/sample == 5 samples ahead -> target == current_value + 5
    row = dataset.iloc[10]
    assert row["target"] == pytest.approx(row["current_value"] + 5)


@pytest.mark.behavioural
def test_tail_rows_without_a_future_value_are_dropped():
    samples = _samples(50, interval=2.0)
    config = FeatureEngineeringConfig(lookback_windows_seconds=(10,), tolerance_seconds=2.0)
    fm = compute_feature_matrix(samples, config)

    dataset = build_supervised_dataset(samples, fm, "cpu_percent", horizon_seconds=20, tolerance_seconds=2.0)
    # Last row's timestamp + 20s has no corresponding sample -> dropped.
    assert len(dataset) < len(fm)
    last_kept_ts = dataset["timestamp"].max()
    assert last_kept_ts <= samples[-1].timestamp - 20 + 2.0  # within tolerance


@pytest.mark.behavioural
def test_current_value_is_exact_match_to_raw_sample():
    samples = _samples(30, interval=2.0)
    config = FeatureEngineeringConfig(lookback_windows_seconds=(10,), tolerance_seconds=2.0)
    fm = compute_feature_matrix(samples, config)
    dataset = build_supervised_dataset(samples, fm, "cpu_percent", horizon_seconds=10, tolerance_seconds=2.0)

    for _, row in dataset.iterrows():
        matching_sample = next(s for s in samples if s.timestamp == row["timestamp"])
        assert row["current_value"] == pytest.approx(matching_sample.cpu_percent)


@pytest.mark.plumbing
def test_unknown_metric_name_raises_key_error():
    samples = _samples(30)
    config = FeatureEngineeringConfig()
    fm = compute_feature_matrix(samples, config)
    with pytest.raises(KeyError):
        build_supervised_dataset(samples, fm, "not_a_real_metric", horizon_seconds=10, tolerance_seconds=2.0)


@pytest.mark.plumbing
def test_empty_feature_matrix_returns_empty_dataset():
    config = FeatureEngineeringConfig()
    result = build_supervised_dataset([], compute_feature_matrix([], config), "cpu_percent", 10, 2.0)
    assert result.empty


@pytest.mark.behavioural
def test_counter_metric_without_is_counter_flag_has_astronomically_large_target():
    """Regression test for a real bug: a counter metric's raw
    cumulative value grows unboundedly, while the features available to
    predict it are all small-magnitude rates. Without is_counter=True,
    target and feature scales are wildly mismatched."""
    samples = [
        SystemMetricSample(timestamp=float(i) * 2.0, cpu_percent=1.0, mem_percent=1.0,
                            disk_read_bytes=1000 * i)  # raw cumulative counter, grows unbounded
        for i in range(100)
    ]
    config = FeatureEngineeringConfig(lookback_windows_seconds=(10, 30), tolerance_seconds=3.0)
    fm = compute_feature_matrix(samples, config)

    without_flag = build_supervised_dataset(samples, fm, "disk_read_bytes", 10, 3.0, is_counter=False)
    with_flag = build_supervised_dataset(samples, fm, "disk_read_bytes", 10, 3.0, is_counter=True)

    # Without the fix: raw counter values in the tens of thousands+.
    assert without_flag["target"].max() > 10_000
    # With the fix: the target is a small, bounded per-second rate (here,
    # a clean 500 bytes/sec — 1000 bytes every 2s).
    assert with_flag["target"].max() < 1_000
    assert with_flag["target"].iloc[10] == pytest.approx(500.0, rel=0.05)


@pytest.mark.behavioural
def test_counter_metric_current_value_is_also_rate_not_raw(
):
    samples = [
        SystemMetricSample(timestamp=float(i) * 2.0, cpu_percent=1.0, mem_percent=1.0,
                            disk_read_bytes=1000 * i)
        for i in range(100)
    ]
    config = FeatureEngineeringConfig(lookback_windows_seconds=(10, 30), tolerance_seconds=3.0)
    fm = compute_feature_matrix(samples, config)
    ds = build_supervised_dataset(samples, fm, "disk_read_bytes", 10, 3.0, is_counter=True)

    # current_value should be on the same small scale as target, not the
    # raw cumulative counter (which by row 50 would be 50,000).
    assert ds["current_value"].max() < 1_000
