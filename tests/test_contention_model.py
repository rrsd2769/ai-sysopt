from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiperfopt.common.exceptions import ContentionPredictionError
from aiperfopt.contention.model import ContentionClassifier


def _separable_training_data(n=100, seed=0):
    """label = 1 iff feature_a is large; feature_b is pure noise (irrelevant)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 100, n)
    b = rng.uniform(0, 100, n)
    label = (a > 70).astype(int)
    return pd.DataFrame({"feature_a": a, "feature_b": b}), pd.Series(label)


@pytest.mark.behavioural
def test_fit_and_predict_proba_separates_classes():
    X, y = _separable_training_data()
    classifier = ContentionClassifier(min_training_rows=20)
    classifier.fit(X, y)

    high_row = pd.Series({"feature_a": 95.0, "feature_b": 50.0})
    low_row = pd.Series({"feature_a": 5.0, "feature_b": 50.0})

    assert classifier.predict_proba(high_row) > classifier.predict_proba(low_row)


@pytest.mark.plumbing
def test_fit_with_too_few_rows_raises():
    X, y = _separable_training_data(n=10)
    classifier = ContentionClassifier(min_training_rows=50)
    with pytest.raises(ContentionPredictionError):
        classifier.fit(X, y)


@pytest.mark.plumbing
def test_fit_with_single_class_raises():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0] * 10})
    y = pd.Series([0] * 30)  # only one class present
    classifier = ContentionClassifier(min_training_rows=10)
    with pytest.raises(ContentionPredictionError):
        classifier.fit(X, y)


@pytest.mark.plumbing
def test_predict_proba_before_fit_raises():
    classifier = ContentionClassifier()
    with pytest.raises(ContentionPredictionError):
        classifier.predict_proba(pd.Series({"a": 1.0}))


@pytest.mark.behavioural
def test_predict_proba_batch_matches_individual_predictions():
    X, y = _separable_training_data()
    classifier = ContentionClassifier(min_training_rows=20)
    classifier.fit(X, y)

    batch = classifier.predict_proba_batch(X.iloc[:5])
    individual = [classifier.predict_proba(X.iloc[i]) for i in range(5)]

    for b, i in zip(batch.to_numpy(), individual):
        assert b == pytest.approx(i, abs=1e-6)


@pytest.mark.behavioural
def test_contributing_metrics_ranks_the_actually_informative_feature_first():
    """feature_a determines the label; feature_b is noise. For a clearly
    high-feature_a row, feature_a should dominate the attribution."""
    X, y = _separable_training_data(n=300)
    classifier = ContentionClassifier(min_training_rows=20)
    classifier.fit(X, y)

    row = pd.Series({"feature_a": 99.0, "feature_b": 50.0})
    contributions = classifier.contributing_metrics(row, top_n=2)

    assert contributions[0].metric_name == "feature_a"


@pytest.mark.plumbing
def test_contributing_metrics_before_fit_raises():
    classifier = ContentionClassifier()
    with pytest.raises(ContentionPredictionError):
        classifier.contributing_metrics(pd.Series({"a": 1.0}))


@pytest.mark.plumbing
def test_predict_proba_handles_missing_feature_gracefully():
    X, y = _separable_training_data()
    classifier = ContentionClassifier(min_training_rows=20)
    classifier.fit(X, y)

    incomplete_row = pd.Series({"feature_a": 50.0})  # feature_b missing
    proba = classifier.predict_proba(incomplete_row)  # must not raise
    assert 0.0 <= proba <= 1.0


@pytest.mark.plumbing
def test_decision_threshold_defaults_to_none_and_is_settable():
    """Unset means "never fitted" -- callers fall back to the
    configured constant in that case (see aiperfopt.api.live_state)."""
    classifier = ContentionClassifier()
    assert classifier.decision_threshold is None

    classifier.decision_threshold = 0.62
    assert classifier.decision_threshold == pytest.approx(0.62)
