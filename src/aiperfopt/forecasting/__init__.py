"""Forecasting Models + Model Evaluation & Validation (Section 7.1, items 5 & 10).

- ``synthetic``          -> synthetic scenario generation (Section 10.3 Phase 1)
- ``dataset``             -> supervised dataset construction (feature matrix + forward-looking label)
- ``baselines``            -> the shared Forecaster protocol + 3 mandatory baselines (Section 10.8)
- ``model``                 -> MetricForecaster, the LightGBM-backed forecaster (ADL-13)
- ``persistence``           -> save/load trained model artifacts
- ``evaluation``             -> walk-forward validation harness (Section 11, FR-10.1)
- ``training_pipeline``       -> the Phase 4 entrypoint: train_and_evaluate_all()
"""

from aiperfopt.forecasting.baselines import (
    ExponentialSmoothingForecaster,
    ForecastResult,
    Forecaster,
    MovingAverageForecaster,
    NaivePersistenceForecaster,
)
from aiperfopt.forecasting.dataset import build_supervised_dataset
from aiperfopt.forecasting.deployment import load_deployed_forecaster
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
from aiperfopt.synthetic import (
    HELD_OUT_SCENARIO_GENERATOR,
    HELD_OUT_SCENARIO_NAME,
    SCENARIO_GENERATORS,
)
from aiperfopt.forecasting.training_pipeline import (
    MetricHorizonOutcome,
    TrainingRunSummary,
    baseline_factories_for,
    generate_holdout_scenario,
    generate_training_scenarios,
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
    "load_deployed_forecaster",
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
    "HELD_OUT_SCENARIO_GENERATOR",
    "HELD_OUT_SCENARIO_NAME",
    "SCENARIO_GENERATORS",
    "MetricHorizonOutcome",
    "TrainingRunSummary",
    "baseline_factories_for",
    "generate_holdout_scenario",
    "generate_training_scenarios",
    "train_and_evaluate_all",
    "train_evaluate_and_select_one",
]
