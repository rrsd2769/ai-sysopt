from __future__ import annotations

import math

import pandas as pd
import pytest

from aiperfopt.common.timeutils import epoch_seconds_to_utc_datetime
from aiperfopt.forecasting.baselines import (
    ExponentialSmoothingForecaster,
    MovingAverageForecaster,
    NaivePersistenceForecaster,
)

pytestmark = pytest.mark.behavioural


def test_naive_persistence_predicts_current_value():
    row = pd.Series({"current_value": 42.5, "timestamp": 100.0})
    result = NaivePersistenceForecaster().predict(row)
    assert result.predicted_value == 42.5
    assert result.confidence is None


def test_moving_average_uses_matching_rolling_column():
    row = pd.Series({"current_value": 10.0, "cpu_percent_roll_mean_30s": 25.0})
    forecaster = MovingAverageForecaster(metric_name="cpu_percent", window_seconds=30)
    result = forecaster.predict(row)
    assert result.predicted_value == 25.0


def test_moving_average_falls_back_to_current_value_when_column_missing():
    row = pd.Series({"current_value": 10.0})
    forecaster = MovingAverageForecaster(metric_name="cpu_percent", window_seconds=30)
    result = forecaster.predict(row)
    assert result.predicted_value == 10.0


def test_moving_average_falls_back_when_column_is_nan():
    row = pd.Series({"current_value": 10.0, "cpu_percent_roll_mean_30s": float("nan")})
    forecaster = MovingAverageForecaster(metric_name="cpu_percent", window_seconds=30)
    result = forecaster.predict(row)
    assert result.predicted_value == 10.0


def _build_history(values, interval=2.0):
    timestamps = [i * interval for i in range(len(values))]
    df = pd.DataFrame({"timestamp": timestamps, "value": values})
    df["_dt"] = epoch_seconds_to_utc_datetime(df["timestamp"])
    return df.set_index("_dt")["value"]


def test_exponential_smoothing_predicts_reasonable_value_on_trending_data():
    values = [float(i) for i in range(30)]  # clean linear trend
    history = _build_history(values)
    row = pd.Series({"current_value": values[-1], "timestamp": (len(values) - 1) * 2.0})

    forecaster = ExponentialSmoothingForecaster(horizon_seconds=10)
    result = forecaster.predict(row, history=history)

    # A clean upward linear trend should be forecast to continue rising.
    assert result.predicted_value > values[-1]


def test_exponential_smoothing_falls_back_with_insufficient_history():
    row = pd.Series({"current_value": 5.0, "timestamp": 10.0})
    forecaster = ExponentialSmoothingForecaster(horizon_seconds=10)
    result = forecaster.predict(row, history=None)
    assert result.predicted_value == 5.0


def test_exponential_smoothing_falls_back_with_too_short_history():
    history = _build_history([1.0, 2.0])  # below the len(history) < 4 threshold
    row = pd.Series({"current_value": 2.0, "timestamp": 2.0})
    forecaster = ExponentialSmoothingForecaster(horizon_seconds=10)
    result = forecaster.predict(row, history=history)
    assert result.predicted_value == 2.0


def test_exponential_smoothing_handles_degenerate_flat_series_without_raising():
    history = _build_history([5.0] * 20)  # perfectly flat -> can be degenerate for Holt
    row = pd.Series({"current_value": 5.0, "timestamp": 38.0})
    forecaster = ExponentialSmoothingForecaster(horizon_seconds=10)
    result = forecaster.predict(row, history=history)  # must not raise
    assert not math.isnan(result.predicted_value)


def test_exponential_smoothing_never_returns_nan_with_nan_contaminated_history():
    """Regression test for a real bug found training on counter-rate
    series spanning concatenated synthetic scenarios: a trailing window
    that includes NaN gaps (e.g. from a scenario-boundary artifact) could
    make statsmodels' Holt silently return NaN without raising, which the
    original try/except never caught."""
    values = [1.0, 2.0, float("nan"), float("nan"), 3.0, 4.0, float("nan"), 5.0, 6.0, 7.0]
    history = _build_history(values)
    row = pd.Series({"current_value": 7.0, "timestamp": 18.0})

    forecaster = ExponentialSmoothingForecaster(horizon_seconds=10)
    result = forecaster.predict(row, history=history)

    assert not math.isnan(result.predicted_value)
    assert not math.isinf(result.predicted_value)


def test_exponential_smoothing_falls_back_when_history_is_entirely_nan():
    values = [float("nan")] * 10
    history = _build_history(values)
    row = pd.Series({"current_value": 42.0, "timestamp": 18.0})

    forecaster = ExponentialSmoothingForecaster(horizon_seconds=10)
    result = forecaster.predict(row, history=history)

    assert result.predicted_value == 42.0  # falls back to current_value
