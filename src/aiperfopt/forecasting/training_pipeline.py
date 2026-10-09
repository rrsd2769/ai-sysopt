"""
Forecasting training pipeline orchestrator.

Ties together the rest of :mod:`aiperfopt.forecasting` into one callable:
compute features per scenario, run walk-forward evaluation against
baselines, persist evaluation results, save the winning artifact, and
check generalization against a held-out slice.

Why features are computed per scenario, not on one concatenated series
----------------------------------------------------------------------
Scenarios are independent runs with no continuity between the end of one
and the start of the next. Computing lag/rolling features on a naive
concatenation would let a rolling window at the start of one scenario
include samples from the tail of the previous one. Each scenario is
feature-computed on its own, and only the resulting labeled feature rows
are concatenated afterward for walk-forward cross-validation.

Why walk-forward CV wants more than one shape of data
-----------------------------------------------------
Running walk-forward CV on a single deterministic series trains early
folds on one regime and tests them on another, which is not a fair
evaluation and made LightGBM look worse than every baseline. Combining
differently-shaped scenarios before running ``TimeSeriesSplit`` gives each
fold's training data variety to learn from.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import Engine

from aiperfopt.applog import get_logger
from aiperfopt.common.timeutils import epoch_seconds_to_utc_datetime
from aiperfopt.config.settings import ForecastingSettings
from aiperfopt.features import FeatureEngineeringConfig, compute_feature_matrix
from aiperfopt.features.engine import counter_rate_per_second
from aiperfopt.forecasting.baselines import (
    ExponentialSmoothingForecaster,
    Forecaster,
    MovingAverageForecaster,
    NaivePersistenceForecaster,
)
from aiperfopt.forecasting.dataset import NON_FEATURE_COLUMNS, build_supervised_dataset
from aiperfopt.forecasting.evaluation import (
    evaluate_forecaster_on_fold,
    mean_metrics_by_model,
    run_walk_forward_evaluation,
    select_best_forecaster_name,
)
from aiperfopt.forecasting.model import MetricForecaster
from aiperfopt.forecasting.persistence import save_forecaster
from aiperfopt.storage.records import EvaluationRecord, SystemMetricSample
from aiperfopt.storage.writer import insert_evaluations_batch

logger = get_logger(__name__)


@dataclass
class MetricHorizonOutcome:
    metric_name: str
    horizon_label: str
    horizon_seconds: int
    walk_forward_summary: dict[str, dict[str, float]]
    selected_forecaster: str
    holdout_mae: float | None
    holdout_rmse: float | None
    holdout_directional_accuracy: float | None
    artifact_path: str | None = None
    is_counter: bool = False
    # The evidence pack (aiperfopt.evaluation.report): mean |SHAP
    # value| per feature over the holdout set, for the winning LightGBM
    # model only. Only populated when the caller opts in via
    # ``compute_feature_importance`` -- computing SHAP on every ordinary
    # training/retraining run would be pure overhead nobody asked for.
    feature_importance: dict[str, float] | None = None
    # The quality gate needs a persistence-baseline number measured
    # on the *same* holdout set the deployed model was scored on (not the
    # walk-forward CV mean, which is a different data slice) -- otherwise
    # "beats persistence" would be comparing two different evaluations.
    # ``NaivePersistenceForecaster`` is stateless, so this costs one extra
    # cheap prediction pass, not a second training run.
    holdout_persistence_mae: float | None = None
    holdout_persistence_directional_accuracy: float | None = None


@dataclass
class TrainingRunSummary:
    model_version: str
    outcomes: list[MetricHorizonOutcome] = field(default_factory=list)


def _combined_raw_history(scenarios: dict[str, list[SystemMetricSample]], metric_name: str) -> pd.Series:
    all_samples = [s for samples in scenarios.values() for s in samples]
    raw_df = pd.DataFrame.from_records(
        [{"timestamp": s.timestamp, metric_name: getattr(s, metric_name)} for s in all_samples]
    )
    raw_df = raw_df.dropna(subset=[metric_name]).sort_values("timestamp")
    raw_df["_dt"] = epoch_seconds_to_utc_datetime(raw_df["timestamp"])
    return raw_df.set_index("_dt")[metric_name].astype(float)


def baseline_factories_for(
    metric_name: str, horizon_seconds: int, is_counter: bool = False
) -> dict[str, "Forecaster"]:
    """Public: also used by :mod:`aiperfopt.lifecycle.deployment` to
    reconstruct a deployed baseline forecaster outside a training run.

    ``is_counter`` matters only for :class:`MovingAverageForecaster`: a
    counter metric's rolling-mean feature column is named
    ``{metric_name}_rate_roll_mean_{window}s`` (Feature Engineering
    rate-transforms counters before computing any rolling statistic —
    see :mod:`aiperfopt.features.engine`), not
    ``{metric_name}_roll_mean_{window}s``. Getting this wrong means the
    baseline silently falls through to its "column not found" fallback
    every time, quietly returning ``current_value`` instead of an actual
    moving average.
    """
    short_window = min(30, horizon_seconds)
    moving_average_column_name = f"{metric_name}_rate" if is_counter else metric_name
    return {
        "naive_persistence": lambda: NaivePersistenceForecaster(),
        "moving_average": lambda: MovingAverageForecaster(
            metric_name=moving_average_column_name, window_seconds=short_window
        ),
        "exponential_smoothing": lambda: ExponentialSmoothingForecaster(horizon_seconds=horizon_seconds),
    }


def _persist_fold_evaluations(
    engine: Engine,
    fold_results,
    metric_name: str,
    horizon_label: str,
    model_version: str,
) -> None:
    records = [
        EvaluationRecord(
            evaluated_at=time.time(),
            model_name="lightgbm" if r.model_name == "lightgbm" else r.model_name,
            model_version=model_version,
            metric_name=metric_name,
            horizon_label=horizon_label,
            fold_index=r.fold_index,
            dataset_tag="walk_forward",
            mae=r.mae,
            rmse=r.rmse,
            directional_accuracy=r.directional_accuracy,
            baseline_name=None if r.model_name == "lightgbm" else r.model_name,
        )
        for r in fold_results
    ]
    if records:
        insert_evaluations_batch(engine, records)


def train_evaluate_and_select_one(
    engine: Engine,
    scenarios: dict[str, list[SystemMetricSample]],
    holdout_samples: list[SystemMetricSample],
    feature_config: FeatureEngineeringConfig,
    forecasting_settings: ForecastingSettings,
    metric_name: str,
    horizon_label: str,
    horizon_seconds: int,
    model_version: str,
    holdout_dataset_tag: str,
    is_counter: bool = False,
    process_summaries: dict[str, pd.DataFrame] | None = None,
    holdout_process_summary: pd.DataFrame | None = None,
    compute_feature_importance: bool = False,
) -> MetricHorizonOutcome:
    """Full pipeline for one (metric, horizon) pair: walk-forward eval, selection,
    held-out generalization check, and artifact persistence if LightGBM wins.

    ``is_counter`` must be set for cumulative OS counters (disk/net I/O
    byte counts) — see :mod:`aiperfopt.forecasting.dataset`'s module
    docstring for why getting this wrong silently breaks both the
    LightGBM target and the moving-average baseline for that metric.

    ``process_summaries`` / ``holdout_process_summary``: optional, keyed
    by scenario name (matching ``scenarios``) / for the holdout set
    respectively — an already-aligned process-summary frame (see
    :func:`aiperfopt.features.storage_bridge.aligned_process_summary`) to
    merge into that scenario's feature matrix. Omitted (the default)
    trains on system metrics only. The ablation
    (:mod:`aiperfopt.evaluation.ablation`) uses this to give the
    "system + process" arm real process data.
    """
    tolerance = feature_config.tolerance_seconds

    per_scenario_datasets = []
    for scenario_name, samples in scenarios.items():
        process_summary = (process_summaries or {}).get(scenario_name)
        fm = compute_feature_matrix(samples, feature_config, process_summary=process_summary)
        ds = build_supervised_dataset(
            samples, fm, metric_name, horizon_seconds, tolerance, is_counter=is_counter
        )
        if not ds.empty:
            per_scenario_datasets.append(ds)

    combined = (
        pd.concat(per_scenario_datasets, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        if per_scenario_datasets
        else pd.DataFrame()
    )

    raw_history = _combined_raw_history(scenarios, metric_name)
    if is_counter:
        # Baselines that consume raw_history (Exponential Smoothing) must
        # also see the rate series, not the raw cumulative counter.
        raw_history = counter_rate_per_second(raw_history, max_gap_seconds=30.0)

    fold_results = run_walk_forward_evaluation(
        combined,
        metric_name=metric_name,
        horizon_label=horizon_label,
        horizon_seconds=horizon_seconds,
        model_version=model_version,
        n_splits=forecasting_settings.walk_forward_splits,
        stability_threshold=forecasting_settings.directional_stability_threshold,
        min_training_rows=forecasting_settings.min_training_rows,
        raw_history=raw_history,
        baseline_factories=baseline_factories_for(metric_name, horizon_seconds, is_counter=is_counter),
    )
    _persist_fold_evaluations(engine, fold_results, metric_name, horizon_label, model_version)

    summary = mean_metrics_by_model(fold_results)
    if not summary:
        logger.warning(
            "No walk-forward folds produced, skipping metric/horizon",
            extra={"metric": metric_name, "horizon": horizon_label},
        )
        return MetricHorizonOutcome(
            metric_name, horizon_label, horizon_seconds, {}, "none", None, None, None,
            is_counter=is_counter,
        )

    selected = select_best_forecaster_name(summary, forecasting_settings.baseline_improvement_margin)

    # Held-out generalization check — the scenario reserved
    # here never appeared in any walk-forward training fold above.
    holdout_fm = compute_feature_matrix(holdout_samples, feature_config, process_summary=holdout_process_summary)
    holdout_dataset = build_supervised_dataset(
        holdout_samples, holdout_fm, metric_name, horizon_seconds, tolerance, is_counter=is_counter
    )

    artifact_path = None
    holdout_mae = holdout_rmse = holdout_directional = None

    feature_cols = [c for c in combined.columns if c not in NON_FEATURE_COLUMNS]

    if selected == "lightgbm":
        final_model = MetricForecaster(
            metric_name=metric_name, horizon_label=horizon_label, horizon_seconds=horizon_seconds,
            model_version=model_version, min_training_rows=forecasting_settings.min_training_rows,
        )
        final_model.fit(combined[feature_cols], combined["target"] - combined["current_value"])
        final_model.set_confidence_from_validation(
            rmse=summary["lightgbm"]["rmse"], target_std=float(combined["target"].std())
        )
        saved_dir = save_forecaster(final_model, forecasting_settings.models_dir)
        artifact_path = str(saved_dir)
        deployed_forecaster: Forecaster = final_model
    else:
        deployed_forecaster = baseline_factories_for(metric_name, horizon_seconds, is_counter=is_counter)[selected]()

    feature_importance = None
    if compute_feature_importance and selected == "lightgbm" and not holdout_dataset.empty:
        feature_importance = _shap_feature_importance(final_model, holdout_dataset[feature_cols])

    holdout_persistence_mae = holdout_persistence_directional = None

    if not holdout_dataset.empty:
        holdout_history = _combined_raw_history({"holdout": holdout_samples}, metric_name)
        if is_counter:
                holdout_history = counter_rate_per_second(holdout_history, max_gap_seconds=30.0)
        holdout_mae, holdout_rmse, holdout_directional = evaluate_forecaster_on_fold(
            deployed_forecaster, holdout_dataset,
            forecasting_settings.directional_stability_threshold,
            raw_history=holdout_history,
        )
        holdout_persistence_mae, _, holdout_persistence_directional = evaluate_forecaster_on_fold(
            NaivePersistenceForecaster(), holdout_dataset,
            forecasting_settings.directional_stability_threshold,
            raw_history=holdout_history,
        )
        insert_evaluations_batch(engine, [
            EvaluationRecord(
                evaluated_at=time.time(),
                model_name=selected, model_version=model_version,
                metric_name=metric_name, horizon_label=horizon_label, fold_index=None,
                dataset_tag=holdout_dataset_tag,
                mae=holdout_mae, rmse=holdout_rmse, directional_accuracy=holdout_directional,
                baseline_name=None if selected == "lightgbm" else selected,
                extra={"selected_forecaster": selected},
            )
        ])

    logger.info(
        "Metric/horizon training complete",
        extra={
            "metric": metric_name, "horizon": horizon_label, "selected": selected,
            "holdout_mae": holdout_mae,
        },
    )

    return MetricHorizonOutcome(
        metric_name=metric_name, horizon_label=horizon_label, horizon_seconds=horizon_seconds,
        walk_forward_summary=summary, selected_forecaster=selected,
        holdout_mae=holdout_mae, holdout_rmse=holdout_rmse,
        holdout_directional_accuracy=holdout_directional, artifact_path=artifact_path,
        is_counter=is_counter, feature_importance=feature_importance,
        holdout_persistence_mae=holdout_persistence_mae,
        holdout_persistence_directional_accuracy=holdout_persistence_directional,
    )


def _shap_feature_importance(model: MetricForecaster, X: pd.DataFrame) -> dict[str, float]:
    """Mean |SHAP value| per feature over ``X`` — the evidence-pack
    feature ranking (expected to show process features ranking highly for
    the "system + process" ablation arm)."""
    import numpy as np
    import shap  # imported lazily: only paid when a caller opts in

    explainer = shap.TreeExplainer(model.booster)
    shap_values = explainer.shap_values(X.astype(float))
    mean_abs = np.abs(shap_values).mean(axis=0)
    return dict(zip(X.columns, (float(v) for v in mean_abs)))


def train_and_evaluate_all(
    engine: Engine,
    feature_config: FeatureEngineeringConfig,
    forecasting_settings: ForecastingSettings,
    model_version: str,
    *,
    scenarios: dict[str, list[SystemMetricSample]],
    holdout_samples: list[SystemMetricSample],
    holdout_dataset_tag: str,
    target_metrics: list[str] | None = None,
    process_summaries: dict[str, pd.DataFrame] | None = None,
    holdout_process_summary: pd.DataFrame | None = None,
    compute_feature_importance: bool = False,
) -> TrainingRunSummary:
    """Train and evaluate every (metric, horizon) combination. The training entrypoint.

    Parameters
    ----------
    target_metrics:
        Metrics to forecast. Defaults to every metric Feature Engineering
        is configured to compute features for (``feature_config.gauge_metrics
        + feature_config.counter_metrics``) — every metric we have
        engineered features for is also worth forecasting.
    scenarios / holdout_samples:
        Training data and a chronologically later held-out slice, as built
        by :mod:`aiperfopt.lifecycle.training_data`.
    holdout_dataset_tag:
        Stored on holdout evaluation rows so evaluation history stays
        accurate about what the holdout data was.
    process_summaries / holdout_process_summary:
        See :func:`train_evaluate_and_select_one`. Omitted (the default)
        reproduces exactly pre-Phase-5 behavior for every existing
        caller.
    """
    metrics = target_metrics or list(feature_config.gauge_metrics) + list(feature_config.counter_metrics)

    horizons = [
        ("short", forecasting_settings.short_horizon_target_seconds),
        ("medium", forecasting_settings.medium_horizon_target_seconds),
    ]

    summary = TrainingRunSummary(model_version=model_version)
    for metric_name in metrics:
        is_counter = metric_name in feature_config.counter_metrics
        for horizon_label, horizon_seconds in horizons:
            outcome = train_evaluate_and_select_one(
                engine, scenarios, holdout_samples, feature_config, forecasting_settings,
                metric_name, horizon_label, horizon_seconds, model_version, is_counter=is_counter,
                holdout_dataset_tag=holdout_dataset_tag,
                process_summaries=process_summaries, holdout_process_summary=holdout_process_summary,
                compute_feature_importance=compute_feature_importance,
            )
            summary.outcomes.append(outcome)

    return summary
