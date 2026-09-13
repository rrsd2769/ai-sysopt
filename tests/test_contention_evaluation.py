from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.contention.evaluation import (
    ContentionFoldMetrics,
    compute_base_rate,
    compute_pr_auc,
    evaluate_threshold_rule_baseline,
    feature_columns_of,
    fit_decision_threshold,
    mean_contention_metrics,
    run_contention_walk_forward_evaluation,
)

pytestmark = pytest.mark.behavioural


def test_feature_columns_of_excludes_timestamp_and_label():
    df = pd.DataFrame({"timestamp": [1.0], "label": [0], "forecast_a": [2.0]})
    assert feature_columns_of(df) == ["forecast_a"]


def test_mean_contention_metrics_aggregates():
    results = [
        ContentionFoldMetrics(0, precision=0.8, recall=0.6, f1=0.68, pr_auc=0.75, n_test_rows=10, n_positive=3),
        ContentionFoldMetrics(1, precision=0.6, recall=0.4, f1=0.48, pr_auc=0.55, n_test_rows=10, n_positive=2),
    ]
    summary = mean_contention_metrics(results)
    assert summary["precision"] == pytest.approx(0.7)
    assert summary["recall"] == pytest.approx(0.5)
    assert summary["pr_auc"] == pytest.approx(0.65)
    assert summary["n_folds"] == 2.0


def test_mean_contention_metrics_empty_input():
    assert mean_contention_metrics([]) == {}


def test_run_walk_forward_evaluation_produces_results_on_learnable_data():
    rng = np.random.default_rng(0)
    n = 300
    a = rng.uniform(0, 100, n)
    dataset = pd.DataFrame({
        "timestamp": np.arange(n) * 2.0,
        "forecast_a": a,
        "label": (a > 80).astype(int),
    })

    results = run_contention_walk_forward_evaluation(
        dataset, decision_threshold=0.5, n_splits=3, min_training_rows=20
    )
    assert len(results) > 0
    assert all(0.0 <= r.precision <= 1.0 for r in results)
    assert all(0.0 <= r.recall <= 1.0 for r in results)


def test_run_walk_forward_evaluation_too_few_rows_returns_empty():
    dataset = pd.DataFrame({"timestamp": [1.0, 2.0], "forecast_a": [1.0, 2.0], "label": [0, 1]})
    results = run_contention_walk_forward_evaluation(
        dataset, decision_threshold=0.5, n_splits=3, min_training_rows=20
    )
    assert results == []


def test_run_walk_forward_evaluation_skips_single_class_folds_without_raising():
    # All-zero labels -> every fold's training data is single-class.
    n = 100
    dataset = pd.DataFrame({
        "timestamp": np.arange(n) * 2.0, "forecast_a": np.arange(n, dtype=float), "label": [0] * n,
    })
    results = run_contention_walk_forward_evaluation(
        dataset, decision_threshold=0.5, n_splits=3, min_training_rows=10
    )
    assert results == []  # every fold skipped, but no exception raised


# --- Phase 5: base rate, PR-AUC, fitted threshold, threshold-rule baseline ---


def test_compute_base_rate():
    assert compute_base_rate(pd.Series([0, 0, 0, 1])) == pytest.approx(0.25)
    assert compute_base_rate(pd.Series([], dtype=int)) != compute_base_rate(pd.Series([], dtype=int))  # nan != nan


def test_compute_pr_auc_single_class_is_nan():
    labels = pd.Series([0, 0, 0, 0])
    probabilities = pd.Series([0.1, 0.2, 0.3, 0.4])
    assert np.isnan(compute_pr_auc(probabilities, labels))


def test_compute_pr_auc_perfect_separation_is_one():
    labels = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.1, 0.2, 0.8, 0.9])
    assert compute_pr_auc(probabilities, labels) == pytest.approx(1.0)


def test_fit_decision_threshold_single_class_falls_back():
    labels = pd.Series([0, 0, 0])
    probabilities = pd.Series([0.1, 0.2, 0.3])
    threshold, metrics = fit_decision_threshold(probabilities, labels)
    assert threshold == 0.5
    assert np.isnan(metrics["f1"])


def test_fit_decision_threshold_picks_a_separating_point():
    rng = np.random.default_rng(0)
    n = 200
    labels = pd.Series((np.arange(n) >= 150).astype(int))  # last 50 positive
    # Probabilities well-correlated with the label, plus noise.
    probabilities = pd.Series(labels.to_numpy() * 0.6 + rng.uniform(0, 0.3, n))
    threshold, metrics = fit_decision_threshold(probabilities, labels)
    assert 0.0 < threshold < 1.0
    assert metrics["f1"] > 0.5  # a well-separated dataset should fit a decent threshold

    predictions = (probabilities >= threshold).astype(int)
    from sklearn.metrics import f1_score

    assert f1_score(labels, predictions) == pytest.approx(metrics["f1"])


def test_evaluate_threshold_rule_baseline_flags_rows_already_over_threshold():
    labeling_config = ContentionLabelingConfig(value_thresholds={"cpu_percent": 70.0})
    dataset = pd.DataFrame({
        "current_cpu_percent": [50.0, 75.0, 90.0, 10.0],
        "label": [0, 1, 1, 0],
    })
    metrics = evaluate_threshold_rule_baseline(dataset, labeling_config)
    # Rule predicts [0, 1, 1, 0] -- matches every label here exactly.
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_evaluate_threshold_rule_baseline_ignores_metrics_not_in_dataset():
    labeling_config = ContentionLabelingConfig(value_thresholds={"mem_percent": 85.0})
    dataset = pd.DataFrame({"current_cpu_percent": [90.0], "label": [1]})
    metrics = evaluate_threshold_rule_baseline(dataset, labeling_config)
    # No current_mem_percent column present -> rule predicts all-0 -> misses the positive.
    assert metrics["recall"] == pytest.approx(0.0)
