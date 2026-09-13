from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine

from aiperfopt.contention.model import ContentionClassifier
from aiperfopt.contention.prediction import _strip_feature_prefix, predict_contention
from aiperfopt.storage import ProcessMetricSample, insert_process_metric_samples_batch

pytestmark = pytest.mark.behavioural


def _classifier_always_predicting(probability: float) -> ContentionClassifier:
    """A classifier trained so predict_proba is close to a fixed value for our test row."""
    rng = np.random.default_rng(0)
    n = 100
    a = rng.uniform(0, 100, n)
    X = pd.DataFrame({"forecast_cpu_percent": a, "current_cpu_percent": a})
    y = pd.Series((a > 50).astype(int))
    classifier = ContentionClassifier(min_training_rows=20)
    classifier.fit(X, y)
    return classifier


def test_strip_feature_prefix():
    assert _strip_feature_prefix("forecast_cpu_percent") == "cpu_percent"
    assert _strip_feature_prefix("current_mem_percent") == "mem_percent"
    assert _strip_feature_prefix("anomaly_score") == "anomaly_score"


def test_anomaly_score_dampens_reported_probability(storage_engine: Engine):
    classifier = _classifier_always_predicting(0.9)
    high_row = pd.Series({"forecast_cpu_percent": 95.0, "current_cpu_percent": 95.0, "anomaly_score": 0.0})

    no_anomaly = predict_contention(
        classifier, storage_engine, high_row, as_of=100.0, decision_threshold=0.5,
        anomaly_dampening_factor=0.5, contributing_processes_top_n=3, model_version="v1",
    )

    high_row_anomalous = high_row.copy()
    high_row_anomalous["anomaly_score"] = 1.0
    with_anomaly = predict_contention(
        classifier, storage_engine, high_row_anomalous, as_of=100.0, decision_threshold=0.5,
        anomaly_dampening_factor=0.5, contributing_processes_top_n=3, model_version="v1",
    )

    assert with_anomaly.probability < no_anomaly.probability


def test_triggered_flag_respects_decision_threshold(storage_engine: Engine):
    classifier = _classifier_always_predicting(0.9)
    # A moderate row (near the training decision boundary of 50) gives a
    # probability that isn't pinned near 0 or 1, so threshold comparisons
    # are actually distinguishing behavior rather than both landing on
    # the same side of an extremely confident prediction.
    row = pd.Series({"forecast_cpu_percent": 55.0, "current_cpu_percent": 55.0, "anomaly_score": 0.0})

    high_threshold_result = predict_contention(
        classifier, storage_engine, row, as_of=100.0, decision_threshold=0.999999,
        anomaly_dampening_factor=0.0, contributing_processes_top_n=3, model_version="v1",
    )
    low_threshold_result = predict_contention(
        classifier, storage_engine, row, as_of=100.0, decision_threshold=0.000001,
        anomaly_dampening_factor=0.0, contributing_processes_top_n=3, model_version="v1",
    )

    assert high_threshold_result.triggered is False
    assert low_threshold_result.triggered is True


def test_contributing_processes_resolved_from_dominant_metric(storage_engine: Engine):
    insert_process_metric_samples_batch(storage_engine, [
        ProcessMetricSample(timestamp=99.0, pid=1, name="stress_test", cpu_percent=99.0),
        ProcessMetricSample(timestamp=99.0, pid=2, name="idle_proc", cpu_percent=1.0),
    ])
    classifier = _classifier_always_predicting(0.9)
    row = pd.Series({"forecast_cpu_percent": 95.0, "current_cpu_percent": 95.0, "anomaly_score": 0.0})

    result = predict_contention(
        classifier, storage_engine, row, as_of=100.0, decision_threshold=0.5,
        anomaly_dampening_factor=0.0, contributing_processes_top_n=2, model_version="v1",
    )

    assert len(result.contributing_processes) > 0
    names = [p["name"] for p in result.contributing_processes]
    assert "stress_test" in names


def test_probability_bounded_to_zero_one(storage_engine: Engine):
    classifier = _classifier_always_predicting(0.9)
    row = pd.Series({"forecast_cpu_percent": 95.0, "current_cpu_percent": 95.0, "anomaly_score": 1.0})

    result = predict_contention(
        classifier, storage_engine, row, as_of=100.0, decision_threshold=0.5,
        anomaly_dampening_factor=1.0, contributing_processes_top_n=3, model_version="v1",
    )
    assert 0.0 <= result.probability <= 1.0


def test_result_carries_model_version_and_anomaly_score(storage_engine: Engine):
    classifier = _classifier_always_predicting(0.9)
    row = pd.Series({"forecast_cpu_percent": 95.0, "current_cpu_percent": 95.0, "anomaly_score": 0.42})

    result = predict_contention(
        classifier, storage_engine, row, as_of=100.0, decision_threshold=0.5,
        anomaly_dampening_factor=0.3, contributing_processes_top_n=3, model_version="v1",
    )
    assert result.model_version == "v1"
    assert result.anomaly_score == 0.42
    assert result.threshold_used == 0.5
