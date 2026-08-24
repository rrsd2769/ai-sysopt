from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiperfopt.contention.evaluation import (
    feature_columns_of,
    mean_contention_metrics,
    run_contention_walk_forward_evaluation,
)
from aiperfopt.contention.evaluation import ContentionFoldMetrics


def test_feature_columns_of_excludes_timestamp_and_label():
    df = pd.DataFrame({"timestamp": [1.0], "label": [0], "forecast_a": [2.0]})
    assert feature_columns_of(df) == ["forecast_a"]


def test_mean_contention_metrics_aggregates():
    results = [
        ContentionFoldMetrics(0, precision=0.8, recall=0.6, f1=0.68, n_test_rows=10, n_positive=3),
        ContentionFoldMetrics(1, precision=0.6, recall=0.4, f1=0.48, n_test_rows=10, n_positive=2),
    ]
    summary = mean_contention_metrics(results)
    assert summary["precision"] == pytest.approx(0.7)
    assert summary["recall"] == pytest.approx(0.5)
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
