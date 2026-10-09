from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiperfopt.features.preprocessing import (
    FeaturePreprocessor,
    clean_supervised_rows,
    segment_ids,
    signed_log1p,
)


def _dataset(timestamps, current=None, target=None):
    n = len(timestamps)
    return pd.DataFrame({
        "timestamp": timestamps,
        "current_value": current if current is not None else [10.0] * n,
        "target": target if target is not None else [11.0] * n,
    })


@pytest.mark.behavioural
def test_segment_ids_split_on_gaps_larger_than_threshold():
    ts = pd.Series([0.0, 1.5, 3.0, 100.0, 101.5, 500.0])
    assert segment_ids(ts, gap_seconds=30).tolist() == [0, 0, 0, 1, 1, 2]


@pytest.mark.behavioural
def test_warmup_rows_after_each_gap_are_dropped():
    # Two segments of 10 samples 1.5 s apart; warm-up of 5 s drops the first 4 of each.
    first = [i * 1.5 for i in range(10)]
    second = [1000 + i * 1.5 for i in range(10)]
    cleaned, report = clean_supervised_rows(_dataset(first + second), gap_seconds=30, warmup_seconds=5)

    assert report.segments == 2
    assert report.warmup_rows == 8
    assert cleaned["timestamp"].tolist() == first[4:] + second[4:]


@pytest.mark.behavioural
def test_invalid_rows_are_dropped_and_counted_separately_from_warmup():
    ts = [i * 1.5 for i in range(6)]
    current = [10.0, np.nan, 10.0, 10.0, 150.0, 10.0]
    target = [11.0, 11.0, -1.0, 11.0, 11.0, 11.0]
    cleaned, report = clean_supervised_rows(_dataset(ts, current, target), gap_seconds=30, warmup_seconds=0)

    assert report.invalid_rows == 3
    assert report.warmup_rows == 0
    assert cleaned["timestamp"].tolist() == [0.0, 4.5, 7.5]


@pytest.mark.behavioural
def test_signed_log1p_is_symmetric_and_zero_preserving():
    out = signed_log1p(pd.DataFrame({"a": [-99.0, 0.0, 99.0]}))
    assert out["a"].tolist() == pytest.approx([-np.log(100), 0.0, np.log(100)])


@pytest.mark.behavioural
def test_preprocessor_logs_counter_features_and_drops_constant_and_redundant_columns():
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    X = pd.DataFrame({
        "current_value": base,
        "cpu_percent_lag_30s": base * 2 + 1,                      # redundant with current_value
        "mem_percent_roll_mean_30s": rng.normal(size=200),
        "swap_percent_roll_min_120s": np.zeros(200),              # constant
        "net_bytes_recv_rate_roll_mean_30s": rng.exponential(1e6, size=200),
    })
    pre = FeaturePreprocessor().fit(X)

    assert pre.log_columns == ["net_bytes_recv_rate_roll_mean_30s"]
    assert pre.constant_columns == ["swap_percent_roll_min_120s"]
    assert pre.redundant_columns == ["cpu_percent_lag_30s"]
    out = pre.transform(X)
    assert list(out.columns) == ["current_value", "mem_percent_roll_mean_30s", "net_bytes_recv_rate_roll_mean_30s"]
    assert out["net_bytes_recv_rate_roll_mean_30s"].max() < 20  # log scale, not ~1e6


@pytest.mark.behavioural
def test_current_value_is_never_dropped_even_when_redundant():
    base = np.arange(50, dtype=float)
    X = pd.DataFrame({"cpu_percent_lag_30s": base, "current_value": base + 1})
    pre = FeaturePreprocessor().fit(X)
    assert "current_value" in pre.kept_columns


@pytest.mark.behavioural
def test_transform_reuses_training_decisions_on_new_rows():
    # A column that is constant in training is dropped even if it varies later:
    # the decision is fitted on training rows only, so holdout data can't change it.
    train = pd.DataFrame({"current_value": np.arange(20.0), "x_roll_std_30s": np.zeros(20)})
    later = pd.DataFrame({"current_value": np.arange(5.0), "x_roll_std_30s": np.arange(5.0)})
    pre = FeaturePreprocessor().fit(train)
    assert list(pre.transform(later).columns) == ["current_value"]


@pytest.mark.behavioural
def test_missing_values_are_left_for_the_model():
    X = pd.DataFrame({"current_value": [1.0, 2.0, 3.0, 4.0], "a_lag_30s": [np.nan, 1.0, 5.0, 2.0]})
    out = FeaturePreprocessor().fit_transform(X)
    assert np.isnan(out["a_lag_30s"].iloc[0])


@pytest.mark.plumbing
def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        FeaturePreprocessor().transform(pd.DataFrame({"current_value": [1.0]}))
