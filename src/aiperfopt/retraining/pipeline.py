"""
The Model Retraining entrypoint (Section 7.1 item 11, FR-11.1). Wraps
Phase 4/5's training pipelines verbatim -- this module contributes no new
training logic of its own, only *where the data comes from* (accumulated
real-usage history instead of a fresh synthetic bootstrap) and *what
happens after training* (comparison against whatever's currently
deployed, producing a candidate for a human to confirm or reject).

Why real-usage retraining reuses Phase 4/5's pipelines unchanged
-------------------------------------------------------------------
``train_and_evaluate_all`` and ``train_contention_and_anomaly_pipeline``
already do everything a retraining run needs: walk-forward validation,
baseline comparison, held-out generalization checking, artifact
persistence. The only assumption baked into their *default* behavior was
"scenarios come from synthetic generation" -- which their new
``scenarios``/``holdout_samples`` parameters (added alongside this
module) now let a caller override. Rebuilding that machinery here would
duplicate a lot of already-tested logic for no behavioral difference.

A single ``model_version`` spans forecasting + contention + anomaly
------------------------------------------------------------------------
Consistent with how Phase 4/5 already work: one retraining run trains
and saves every forecasting model, the contention classifier, and the
anomaly detector under the *same* ``model_version`` string. Deployment
(:mod:`aiperfopt.retraining.deployment`) therefore also activates or
rolls back all three together as one unit, never independently -- partial
activation (e.g. a new forecaster paired with an old contention
classifier trained against a different forecaster's output distribution)
is not a state this system is designed to reason about.
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
from aiperfopt.retraining.comparison import RetrainingComparison, compare_candidate_to_active
from aiperfopt.retraining.deployment import get_active_model_version
from aiperfopt.retraining.real_usage_data import REAL_USAGE_SCENARIO_NAME, build_real_usage_training_data
from aiperfopt.storage.records import SystemMetricSample

logger = get_logger(__name__)


@dataclass
class RetrainingCandidate:
    """Everything a human needs to decide whether to deploy this candidate."""

    model_version: str
    data_source: str  # "real_usage" | "synthetic"
    forecast_summary: TrainingRunSummary
    contention_summary: ContentionTrainingSummary | None
    comparison: RetrainingComparison


def _short_horizon_outcomes(forecast_summary: TrainingRunSummary) -> dict:
    """Outcomes usable as Contention Prediction inputs (Phase 5's own constraint).

    Only the short horizon is used for contention (see
    :mod:`aiperfopt.contention.training_pipeline`'s module docstring),
    and only outcomes where a forecaster was actually selected --
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
    data_source: str,
    scenarios: dict[str, list[SystemMetricSample]],
    holdout_samples: list[SystemMetricSample],
    holdout_dataset_tag: str | None,
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

    return RetrainingCandidate(
        model_version=new_model_version, data_source=data_source,
        forecast_summary=forecast_summary, contention_summary=contention_summary, comparison=comparison,
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
    """Retrain against accumulated real-usage history (Section 10.3 Phase 2/3, FR-11.1).

    This is the primary Phase 8 entrypoint -- the one a scheduled or
    manually-triggered retraining job (FR-11.1) is expected to call once
    enough real-usage data has accumulated. Raises
    :class:`aiperfopt.common.exceptions.RetrainingError` (via
    :func:`aiperfopt.retraining.real_usage_data.build_real_usage_training_data`)
    if there isn't yet.
    """
    scenarios, holdout_samples = build_real_usage_training_data(
        engine, start=start, end=end,
        holdout_fraction=retraining_settings.real_usage_holdout_fraction,
        min_total_rows=forecasting_settings.min_training_rows,
    )
    return _run_retraining(
        engine, feature_config, forecasting_settings, contention_settings, labeling_config,
        retraining_settings, new_model_version, data_source=REAL_USAGE_SCENARIO_NAME,
        scenarios=scenarios, holdout_samples=holdout_samples,
        holdout_dataset_tag=f"holdout_{REAL_USAGE_SCENARIO_NAME}", target_metrics=target_metrics,
    )


def run_retraining_on_synthetic_data(
    engine: Engine,
    feature_config: FeatureEngineeringConfig,
    forecasting_settings: ForecastingSettings,
    contention_settings: ContentionSettings,
    labeling_config: ContentionLabelingConfig,
    retraining_settings: RetrainingSettings,
    new_model_version: str,
    target_metrics: list[str] | None = None,
    scenario_duration_seconds: float = 900.0,
    interval_seconds: float = 2.0,
) -> RetrainingCandidate:
    """Retrain against a fresh synthetic bootstrap, the same data source Phase 4/5 used initially.

    Not the primary Phase 8 use case (see this module's docstring and
    Section 10.3) but kept available: it lets retraining -- including the
    comparison-against-active-version and human-confirmation machinery --
    be exercised and demonstrated before real-usage data has accumulated
    enough to retrain on (R-3's mitigation: "V1 does not depend on Phase
    2 data maturing").
    """
    from aiperfopt.forecasting.training_pipeline import generate_holdout_scenario, generate_training_scenarios

    scenarios = generate_training_scenarios(scenario_duration_seconds, interval_seconds)
    holdout_samples = generate_holdout_scenario(
        scenario_duration_seconds, interval_seconds, scenario_count=len(scenarios)
    )
    return _run_retraining(
        engine, feature_config, forecasting_settings, contention_settings, labeling_config,
        retraining_settings, new_model_version, data_source="synthetic",
        scenarios=scenarios, holdout_samples=holdout_samples,
        holdout_dataset_tag=None, target_metrics=target_metrics,
    )
