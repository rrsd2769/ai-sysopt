from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiperfopt.forecasting.evaluation import (
    FoldMetrics,
    compute_directional_accuracy,
    compute_mae_rmse,
    feature_columns_of,
    mean_metrics_by_model,
    run_walk_forward_evaluation,
    select_best_forecaster_name,
)

pytestmark = pytest.mark.behavioural


def test_compute_mae_rmse_known_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    mae, rmse = compute_mae_rmse(y_true, y_pred)
    assert mae == pytest.approx((2 + 2 + 3) / 3)
    assert rmse == pytest.approx(np.sqrt((4 + 4 + 9) / 3))


def test_compute_mae_rmse_perfect_prediction_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    mae, rmse = compute_mae_rmse(y, y)
    assert mae == 0.0
    assert rmse == 0.0


def test_directional_accuracy_all_correct_rising():
    current = np.array([10.0, 10.0, 10.0])
    y_true = np.array([15.0, 15.0, 15.0])  # actually rose
    y_pred = np.array([14.0, 16.0, 20.0])  # predicted rise too
    acc = compute_directional_accuracy(current, y_true, y_pred, stability_threshold=0.02)
    assert acc == 1.0


def test_directional_accuracy_detects_stable_correctly():
    current = np.array([100.0])
    y_true = np.array([100.5])  # 0.5% change -> "stable" under 2% threshold
    y_pred = np.array([100.3])  # also stable
    acc = compute_directional_accuracy(current, y_true, y_pred, stability_threshold=0.02)
    assert acc == 1.0


def test_directional_accuracy_mismatch_scored_correctly():
    current = np.array([10.0])
    y_true = np.array([20.0])  # rose
    y_pred = np.array([5.0])  # predicted fall
    acc = compute_directional_accuracy(current, y_true, y_pred, stability_threshold=0.02)
    assert acc == 0.0


def test_directional_accuracy_empty_input_returns_nan():
    acc = compute_directional_accuracy(np.array([]), np.array([]), np.array([]), 0.02)
    assert np.isnan(acc)


def test_mean_metrics_by_model_aggregates_correctly():
    results = [
        FoldMetrics(0, "lightgbm", mae=2.0, rmse=3.0, directional_accuracy=0.6, n_test_rows=10),
        FoldMetrics(1, "lightgbm", mae=4.0, rmse=5.0, directional_accuracy=0.8, n_test_rows=10),
        FoldMetrics(0, "naive", mae=10.0, rmse=12.0, directional_accuracy=0.1, n_test_rows=10),
    ]
    summary = mean_metrics_by_model(results)
    assert summary["lightgbm"]["mae"] == pytest.approx(3.0)
    assert summary["lightgbm"]["n_folds"] == 2
    assert summary["naive"]["mae"] == pytest.approx(10.0)


def test_select_best_forecaster_lightgbm_wins_with_clear_margin():
    summary = {
        "lightgbm": {"mae": 1.0}, "naive": {"mae": 5.0}, "moving_average": {"mae": 4.0},
    }
    assert select_best_forecaster_name(summary, improvement_margin=0.05) == "lightgbm"


def test_select_best_forecaster_falls_back_to_baseline_when_not_meaningfully_better():
    summary = {
        "lightgbm": {"mae": 4.9}, "naive": {"mae": 5.0}, "moving_average": {"mae": 5.2},
    }
    # lightgbm is barely better than naive (2% improvement) < 5% margin -> falls back
    assert select_best_forecaster_name(summary, improvement_margin=0.05) == "naive"


def test_select_best_forecaster_picks_best_baseline_when_lightgbm_worse():
    summary = {
        "lightgbm": {"mae": 10.0}, "naive": {"mae": 5.0}, "moving_average": {"mae": 4.0},
    }
    assert select_best_forecaster_name(summary, improvement_margin=0.05) == "moving_average"


def test_select_best_forecaster_no_baselines_defaults_to_lightgbm():
    summary = {"lightgbm": {"mae": 10.0}}
    assert select_best_forecaster_name(summary, improvement_margin=0.05) == "lightgbm"


def test_feature_columns_of_excludes_metadata_columns():
    # Phase 5: current_value is a real feature now, only timestamp/target
    # are metadata.
    df = pd.DataFrame({"timestamp": [1.0], "current_value": [2.0], "target": [3.0], "feat_a": [4.0]})
    assert feature_columns_of(df) == ["current_value", "feat_a"]


def test_run_walk_forward_evaluation_produces_folds_for_all_models():
    # Small, clean synthetic dataset with a learnable linear relationship.
    rng = np.random.default_rng(0)
    n = 100
    feat = rng.uniform(0, 10, n)
    dataset = pd.DataFrame({
        "timestamp": np.arange(n) * 2.0,
        "current_value": feat,
        "target": feat * 1.5,
        "feat_a": feat,
    })

    from aiperfopt.forecasting.baselines import NaivePersistenceForecaster

    results = run_walk_forward_evaluation(
        dataset, metric_name="x", horizon_label="short", horizon_seconds=10,
        model_version="test-v1", n_splits=3, stability_threshold=0.02, min_training_rows=10,
        baseline_factories={"naive_persistence": lambda: NaivePersistenceForecaster()},
    )
    model_names = {r.model_name for r in results}
    assert "lightgbm" in model_names
    assert "naive_persistence" in model_names
    assert len(results) > 0


def test_run_walk_forward_evaluation_too_few_rows_returns_empty():
    dataset = pd.DataFrame({
        "timestamp": [1.0, 2.0], "current_value": [1.0, 2.0], "target": [1.0, 2.0], "feat_a": [1.0, 2.0],
    })
    results = run_walk_forward_evaluation(
        dataset, metric_name="x", horizon_label="short", horizon_seconds=10,
        model_version="test-v1", n_splits=3, stability_threshold=0.02, min_training_rows=20,
    )
    assert results == []
