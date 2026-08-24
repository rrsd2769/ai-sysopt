from __future__ import annotations

import pandas as pd
import pytest

from aiperfopt.common.exceptions import ForecastingError
from aiperfopt.forecasting.model import MetricForecaster


def _synthetic_training_frame(n: int = 60):
    # Simple learnable relationship: target = 2*feature_a + feature_b
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.uniform(0, 10, n)
    b = rng.uniform(0, 10, n)
    y = 2 * a + b + rng.normal(0, 0.1, n)
    X = pd.DataFrame({"feature_a": a, "feature_b": b})
    return X, pd.Series(y)


def test_fit_and_predict_learns_a_simple_relationship():
    X, y = _synthetic_training_frame()
    model = MetricForecaster("cpu_percent", "short", 45, "test-v1", min_training_rows=10)
    model.fit(X, y)

    row = pd.Series({"feature_a": 5.0, "feature_b": 5.0})
    result = model.predict(row)
    # True relationship: 2*5 + 5 = 15
    assert abs(result.predicted_value - 15.0) < 3.0


def test_predict_before_fit_raises():
    model = MetricForecaster("cpu_percent", "short", 45, "test-v1")
    with pytest.raises(ForecastingError):
        model.predict(pd.Series({"feature_a": 1.0}))


def test_fit_with_too_few_rows_raises():
    X = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0, 3.0])
    model = MetricForecaster("cpu_percent", "short", 45, "test-v1", min_training_rows=10)
    with pytest.raises(ForecastingError):
        model.fit(X, y)


def test_predict_handles_missing_feature_columns_gracefully():
    """A row missing a column the model was trained on (e.g. a metric
    source that degraded that cycle) should not crash prediction —
    LightGBM handles NaN natively."""
    X, y = _synthetic_training_frame()
    model = MetricForecaster("cpu_percent", "short", 45, "test-v1", min_training_rows=10)
    model.fit(X, y)

    incomplete_row = pd.Series({"feature_a": 5.0})  # feature_b missing entirely
    result = model.predict(incomplete_row)  # must not raise
    assert isinstance(result.predicted_value, float)


def test_predict_batch_matches_single_row_predict():
    X, y = _synthetic_training_frame()
    model = MetricForecaster("cpu_percent", "short", 45, "test-v1", min_training_rows=10)
    model.fit(X, y)

    batch_preds = model.predict_batch(X.iloc[:5])
    single_preds = [model.predict(X.iloc[i]).predicted_value for i in range(5)]

    for batch_val, single_val in zip(batch_preds.to_numpy(), single_preds):
        assert batch_val == pytest.approx(single_val)


def test_set_confidence_from_validation_bounded_zero_to_one():
    model = MetricForecaster("cpu_percent", "short", 45, "test-v1")

    model.set_confidence_from_validation(rmse=0.0, target_std=10.0)
    assert model.confidence == 1.0

    model.set_confidence_from_validation(rmse=100.0, target_std=10.0)  # error >> std
    assert model.confidence == 0.0

    model.set_confidence_from_validation(rmse=5.0, target_std=10.0)
    assert 0.0 < model.confidence < 1.0


def test_set_confidence_handles_zero_target_std():
    model = MetricForecaster("cpu_percent", "short", 45, "test-v1")
    model.set_confidence_from_validation(rmse=1.0, target_std=0.0)
    assert model.confidence == 0.5  # documented degenerate-case fallback


def test_predict_confidence_matches_model_scalar():
    X, y = _synthetic_training_frame()
    model = MetricForecaster("cpu_percent", "short", 45, "test-v1", min_training_rows=10)
    model.fit(X, y)
    model.set_confidence_from_validation(rmse=1.0, target_std=5.0)

    result = model.predict(X.iloc[0])
    assert result.confidence == model.confidence
