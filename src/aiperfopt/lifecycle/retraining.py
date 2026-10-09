"""
The model retraining entrypoint. Wraps the forecasting and contention
training pipelines unchanged: this module contributes only *where the
data comes from* (accumulated real-usage history) and *what happens after
training* (comparison against whatever is currently deployed, producing a
candidate for a human to confirm or reject).

A single ``model_version`` spans forecasting + contention + anomaly
-------------------------------------------------------------------
One retraining run trains and saves every forecasting model, the
contention classifier, and the anomaly detector under the *same*
``model_version`` string. Deployment (:mod:`aiperfopt.lifecycle.deployment`)
therefore also activates or rolls back all three together, never
independently: a new forecaster paired with an old contention classifier
trained against a different forecaster's output distribution is not a
state this system is designed to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine

from aiperfopt.applog import get_logger
from aiperfopt.config.settings import ContentionSettings, ForecastingSettings, RetrainingSettings
from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.contention.training_pipeline import (
    ContentionTrainingSummary,
    train_contention_and_anomaly_pipeline,
)
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.forecasting.training_pipeline import TrainingRunSummary, train_and_evaluate_all
from aiperfopt.lifecycle.comparison import RetrainingComparison, compare_candidate_to_active
from aiperfopt.lifecycle.deployment import get_active_model_version
from aiperfopt.lifecycle.gate import SkillGateResult, evaluate_skill_gate
from aiperfopt.lifecycle.training_data import REAL_USAGE_SCENARIO_NAME, build_real_usage_training_data
from aiperfopt.storage.records import SystemMetricSample

logger = get_logger(__name__)


@dataclass
class RetrainingCandidate:
    """Everything a human needs to decide whether to deploy this candidate."""

    model_version: str
    data_source: str  # always "real_usage"; echoed by the API
    forecast_summary: TrainingRunSummary
    contention_summary: ContentionTrainingSummary | None
    comparison: RetrainingComparison
    skill_gate: SkillGateResult


def _short_horizon_outcomes(forecast_summary: TrainingRunSummary) -> dict:
    """Outcomes usable as Contention Prediction inputs.

    Only the short horizon is used for contention, and only outcomes where a forecaster was actually selected --
    ``selected_forecaster == "none"`` means that metric/horizon produced
    no usable model at all (e.g. too little data), so it cannot feed
    Contention Prediction.
    """
    return {
        o.metric_name: o
        for o in forecast_summary.outcomes
        if o.horizon_label == "short" and o.selected_forecaster != "none"
    }


def _run_retraining(
    engine: Engine,
    feature_config: FeatureEngineeringConfig,
    forecasting_settings: ForecastingSettings,
    contention_settings: ContentionSettings,
    labeling_config: ContentionLabelingConfig,
    retraining_settings: RetrainingSettings,
    new_model_version: str,
    scenarios: dict[str, list[SystemMetricSample]],
    holdout_samples: list[SystemMetricSample],
    holdout_dataset_tag: str,
    target_metrics: list[str] | None,
) -> RetrainingCandidate:
    forecast_summary = train_and_evaluate_all(
        engine, feature_config, forecasting_settings, model_version=new_model_version,
        target_metrics=target_metrics, scenarios=scenarios, holdout_samples=holdout_samples,
        holdout_dataset_tag=holdout_dataset_tag,
    )

    short_outcomes = _short_horizon_outcomes(forecast_summary)
    contention_summary: ContentionTrainingSummary | None = None
    if short_outcomes:
        contention_summary = train_contention_and_anomaly_pipeline(
            engine, feature_config, forecasting_settings, contention_settings, labeling_config,
            short_outcomes, model_version=new_model_version, scenarios=scenarios,
            holdout_samples=holdout_samples, holdout_dataset_tag=holdout_dataset_tag,
        )
    else:
        logger.warning(
            "No short-horizon forecasters trained successfully -- skipping "
            "Contention Prediction/Anomaly Detection retraining",
            extra={"model_version": new_model_version},
        )

    active_version = get_active_model_version(retraining_settings)
    comparison = compare_candidate_to_active(
        engine, new_model_version, forecast_summary, contention_summary,
        active_version, retraining_settings.regression_tolerance,
    )
    skill_gate = evaluate_skill_gate(
        forecast_summary, contention_summary,
        retraining_settings.forecasting_mae_margin, retraining_settings.directional_accuracy_margin,
        retraining_settings.contention_f1_margin,
    )

    return RetrainingCandidate(
        model_version=new_model_version, data_source=REAL_USAGE_SCENARIO_NAME,
        forecast_summary=forecast_summary, contention_summary=contention_summary,
        comparison=comparison, skill_gate=skill_gate,
    )


def run_retraining_on_real_usage_data(
    engine: Engine,
    feature_config: FeatureEngineeringConfig,
    forecasting_settings: ForecastingSettings,
    contention_settings: ContentionSettings,
    labeling_config: ContentionLabelingConfig,
    retraining_settings: RetrainingSettings,
    new_model_version: str,
    target_metrics: list[str] | None = None,
    start: float | None = None,
    end: float | None = None,
) -> RetrainingCandidate:
    """Retrain against accumulated real-usage history.

    Raises
    :class:`aiperfopt.common.exceptions.RetrainingError` (via
    :func:`aiperfopt.lifecycle.training_data.build_real_usage_training_data`)
    if there isn't yet.
    """
    scenarios, holdout_samples = build_real_usage_training_data(
        engine, start=start, end=end,
        holdout_fraction=retraining_settings.real_usage_holdout_fraction,
        min_total_rows=forecasting_settings.min_training_rows,
    )
    return _run_retraining(
        engine, feature_config, forecasting_settings, contention_settings, labeling_config,
        retraining_settings, new_model_version,
        scenarios=scenarios, holdout_samples=holdout_samples,
        holdout_dataset_tag=f"holdout_{REAL_USAGE_SCENARIO_NAME}", target_metrics=target_metrics,
    )

