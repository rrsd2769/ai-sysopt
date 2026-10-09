"""
The ablation: does giving the model process-level leading
indicators actually help predict contention, or not?

Runs the *same* real-usage data and chronological split through the
training pipelines twice -- once with system-only features ("v1", what
the model had before the pivot), once with system+process features
("v2", the pivot) -- and reports the delta. This runs against **real
accumulated usage data**, since process data only exists for recorded
sessions.

Threads a process summary into the training pipelines via their
``process_summaries``/``holdout_process_summary`` parameters -- see
:mod:`aiperfopt.forecasting.training_pipeline` and
:mod:`aiperfopt.contention.training_pipeline`. This answers, for the
ablation specifically, whether real-usage training should ever see
process features; it does
**not** change what a real retraining run (:mod:`aiperfopt.lifecycle.
pipeline`) deploys by default -- that stays v1-shaped pending this
ablation's actual result.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import Engine

from aiperfopt.applog import get_logger
from aiperfopt.config.settings import ContentionSettings, ForecastingSettings
from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.contention.training_pipeline import (
    ContentionTrainingSummary,
    train_contention_and_anomaly_pipeline,
)
from aiperfopt.features.config import SYSTEM_GAUGE_METRICS, FeatureEngineeringConfig
from aiperfopt.features.storage_bridge import aligned_process_summary
from aiperfopt.forecasting.training_pipeline import TrainingRunSummary, train_and_evaluate_all
from aiperfopt.lifecycle.training_data import REAL_USAGE_SCENARIO_NAME, build_real_usage_training_data

logger = get_logger(__name__)

ARM_NAMES = ("v1", "v2")


def feature_config_for_arm(arm: str, base_config: FeatureEngineeringConfig) -> FeatureEngineeringConfig:
    """"v1" = system-only gauges (pre-Phase-4); "v2" = ``base_config`` as given
    (system + process, today's default)."""
    if arm == "v1":
        return dataclasses.replace(base_config, gauge_metrics=SYSTEM_GAUGE_METRICS)
    if arm == "v2":
        return base_config
    raise ValueError(f"Unknown ablation arm {arm!r}, expected one of {ARM_NAMES}")


@dataclass
class ArmResult:
    arm: str  # "v1" | "v2"
    model_version: str
    forecast_summary: TrainingRunSummary
    contention_summary: ContentionTrainingSummary | None


@dataclass
class AblationReport:
    dataset_summary: dict = field(default_factory=dict)
    arms: dict[str, ArmResult] = field(default_factory=dict)


def _short_horizon_outcomes(forecast_summary: TrainingRunSummary) -> dict:
    """Mirrors aiperfopt.lifecycle.retraining's identically-named helper:
    only short-horizon, successfully-selected forecasters feed Contention
    Prediction."""
    return {
        o.metric_name: o
        for o in forecast_summary.outcomes
        if o.horizon_label == "short" and o.selected_forecaster != "none"
    }


def _run_one_arm(
    engine: Engine,
    arm: str,
    base_feature_config: FeatureEngineeringConfig,
    forecasting_settings: ForecastingSettings,
    contention_settings: ContentionSettings,
    labeling_config: ContentionLabelingConfig,
    target_metrics: list[str],
    model_version: str,
    scenarios: dict,
    holdout_samples: list,
    train_process_summary: pd.DataFrame | None,
    holdout_process_summary: pd.DataFrame | None,
) -> ArmResult:
    arm_config = feature_config_for_arm(arm, base_feature_config)
    process_summaries = (
        {REAL_USAGE_SCENARIO_NAME: train_process_summary} if arm == "v2" and train_process_summary is not None
        else None
    )
    arm_holdout_process_summary = holdout_process_summary if arm == "v2" else None

    forecast_summary = train_and_evaluate_all(
        engine, arm_config, forecasting_settings, model_version=model_version,
        target_metrics=target_metrics, scenarios=scenarios, holdout_samples=holdout_samples,
        holdout_dataset_tag=f"ablation_{arm}",
        process_summaries=process_summaries, holdout_process_summary=arm_holdout_process_summary,
        compute_feature_importance=True,
    )

    contention_summary: ContentionTrainingSummary | None = None
    short_outcomes = _short_horizon_outcomes(forecast_summary)
    if short_outcomes:
        contention_summary = train_contention_and_anomaly_pipeline(
            engine, arm_config, forecasting_settings, contention_settings, labeling_config,
            short_outcomes, model_version=model_version, scenarios=scenarios,
            holdout_samples=holdout_samples, holdout_dataset_tag=f"ablation_{arm}",
            process_summaries=process_summaries, holdout_process_summary=arm_holdout_process_summary,
        )
    else:
        logger.warning(
            "No short-horizon forecasters trained successfully for ablation arm -- "
            "skipping Contention Prediction/Anomaly Detection for this arm",
            extra={"arm": arm, "model_version": model_version},
        )

    return ArmResult(
        arm=arm, model_version=model_version,
        forecast_summary=forecast_summary, contention_summary=contention_summary,
    )


def run_ablation(
    engine: Engine,
    feature_config: FeatureEngineeringConfig,
    forecasting_settings: ForecastingSettings,
    contention_settings: ContentionSettings,
    labeling_config: ContentionLabelingConfig,
    target_metrics: list[str],
    model_version_prefix: str,
    holdout_fraction: float = 0.2,
    min_total_rows: int = 1,
    since: float | None = None,
) -> AblationReport:
    """Run both arms (v1: system-only, v2: system+process) on identical
    real-usage data and an identical chronological split.

    Raises :class:`aiperfopt.common.exceptions.RetrainingError` (via
    :func:`aiperfopt.lifecycle.training_data.build_real_usage_training_data`)
    if there isn't yet enough accumulated real-usage data -- surfaced to
    the caller.
    """
    scenarios, holdout_samples = build_real_usage_training_data(
        engine, start=since, holdout_fraction=holdout_fraction, min_total_rows=min_total_rows,
    )
    training_samples = scenarios[REAL_USAGE_SCENARIO_NAME]

    train_process_summary = aligned_process_summary(engine, training_samples, feature_config.tolerance_seconds)
    holdout_process_summary = aligned_process_summary(engine, holdout_samples, feature_config.tolerance_seconds)

    all_samples = training_samples + holdout_samples

    dataset_summary = {
        "training_rows": len(training_samples),
        "holdout_rows": len(holdout_samples),
        "total_rows": len(all_samples),
        "start_timestamp": training_samples[0].timestamp,
        "end_timestamp": holdout_samples[-1].timestamp,
        "hours_spanned": (holdout_samples[-1].timestamp - training_samples[0].timestamp) / 3600.0,
    }

    arms: dict[str, ArmResult] = {}
    for arm in ARM_NAMES:
        model_version = f"{model_version_prefix}-{arm}"
        arms[arm] = _run_one_arm(
            engine, arm, feature_config, forecasting_settings, contention_settings, labeling_config,
            target_metrics, model_version, scenarios, holdout_samples,
            train_process_summary, holdout_process_summary,
        )

    # Labels don't depend on the feature arm (they're derived from actual
    # future raw metric values, not from any engineered feature) -- v2's
    # base rate is exactly v1's too, just read off whichever arm actually
    # trained a classifier.
    v2_contention = arms["v2"].contention_summary
    if v2_contention is not None and v2_contention.base_rate is not None:
        dataset_summary["contention_base_rate"] = v2_contention.base_rate

    return AblationReport(dataset_summary=dataset_summary, arms=arms)
