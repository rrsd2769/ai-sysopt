"""Forecasting Models + Model Evaluation & Validation.

- ``dataset``             -> supervised dataset construction (feature matrix + forward-looking label)
- ``baselines``            -> the shared Forecaster protocol + 3 mandatory baselines
- ``model``                 -> MetricForecaster, the LightGBM-backed forecaster
- ``persistence``           -> save/load trained model artifacts
- ``evaluation``             -> walk-forward validation harness
- ``training_pipeline`` -> the entrypoint: train_and_evaluate_all
"""

from aiperfopt.forecasting.baselines import (
    ExponentialSmoothingForecaster,
    ForecastResult,
    Forecaster,
    MovingAverageForecaster,
    NaivePersistenceForecaster,
)
from aiperfopt.forecasting.dataset import build_supervised_dataset
from aiperfopt.forecasting.evaluation import (
    FoldMetrics,
    build_history_lookup,
    compute_directional_accuracy,
    compute_mae_rmse,
    evaluate_forecaster_on_fold,
    mean_metrics_by_model,
    run_walk_forward_evaluation,
    select_best_forecaster_name,
)
from aiperfopt.forecasting.model import MetricForecaster
from aiperfopt.forecasting.persistence import artifact_exists, load_forecaster, save_forecaster
from aiperfopt.forecasting.training_pipeline import (
    MetricHorizonOutcome,
    TrainingRunSummary,
    baseline_factories_for,
    train_and_evaluate_all,
    train_evaluate_and_select_one,
)

__all__ = [
    "ExponentialSmoothingForecaster",
    "ForecastResult",
    "Forecaster",
    "MovingAverageForecaster",
    "NaivePersistenceForecaster",
    "build_supervised_dataset",
    "FoldMetrics",
    "build_history_lookup",
    "compute_directional_accuracy",
    "compute_mae_rmse",
    "evaluate_forecaster_on_fold",
    "mean_metrics_by_model",
    "run_walk_forward_evaluation",
    "select_best_forecaster_name",
    "MetricForecaster",
    "artifact_exists",
    "load_forecaster",
    "save_forecaster",
    "MetricHorizonOutcome",
    "TrainingRunSummary",
    "baseline_factories_for",
    "train_and_evaluate_all",
    "train_evaluate_and_select_one",
]
