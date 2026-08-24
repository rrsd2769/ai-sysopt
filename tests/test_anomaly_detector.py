from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiperfopt.anomaly.detector import AnomalyDetector
from aiperfopt.common.exceptions import AnomalyDetectionError


def _normal_data(n=100, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.normal(10, 1, n),
        "b": rng.normal(50, 5, n),
    })


def test_fit_with_too_few_rows_raises():
    detector = AnomalyDetector(min_training_rows=50)
    with pytest.raises(AnomalyDetectionError):
        detector.fit(_normal_data(n=10))


def test_score_before_fit_raises():
    detector = AnomalyDetector()
    with pytest.raises(AnomalyDetectionError):
        detector.score(pd.Series({"a": 1.0, "b": 2.0}))


def test_score_returns_bounded_value():
    detector = AnomalyDetector(min_training_rows=20)
    detector.fit(_normal_data())
    score = detector.score(pd.Series({"a": 10.0, "b": 50.0}))
    assert 0.0 <= score <= 1.0


def test_typical_point_scores_lower_than_extreme_outlier():
    detector = AnomalyDetector(min_training_rows=20)
    detector.fit(_normal_data(n=200))

    typical = detector.score(pd.Series({"a": 10.0, "b": 50.0}))
    outlier = detector.score(pd.Series({"a": 1000.0, "b": -1000.0}))

    assert outlier > typical


def test_score_batch_matches_individual_scores():
    detector = AnomalyDetector(min_training_rows=20)
    data = _normal_data(n=100)
    detector.fit(data)

    batch_scores = detector.score_batch(data.iloc[:5])
    individual_scores = [detector.score(data.iloc[i]) for i in range(5)]

    for b, i in zip(batch_scores.to_numpy(), individual_scores):
        assert b == pytest.approx(i, abs=1e-6)


def test_score_handles_missing_columns_via_fillna():
    detector = AnomalyDetector(min_training_rows=20)
    detector.fit(_normal_data())
    incomplete_row = pd.Series({"a": 10.0})  # missing 'b' entirely
    score = detector.score(incomplete_row)  # must not raise
    assert 0.0 <= score <= 1.0


def test_score_handles_nan_values():
    detector = AnomalyDetector(min_training_rows=20)
    detector.fit(_normal_data())
    row_with_nan = pd.Series({"a": float("nan"), "b": 50.0})
    score = detector.score(row_with_nan)  # must not raise
    assert 0.0 <= score <= 1.0


def test_is_significant():
    detector = AnomalyDetector()
    assert detector.is_significant(0.8, threshold=0.7) is True
    assert detector.is_significant(0.5, threshold=0.7) is False
