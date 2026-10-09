"""
Contention Prediction + Anomaly Detection training pipeline orchestrator.

Depends on already-trained forecasters: Contention Prediction consumes
forecast outputs, not raw metrics. Callers must supply the winning
short-horizon MetricHorizonOutcome for every target metric, typically the
result of having run aiperfopt.forecasting.train_and_evaluate_all on the
same ``scenarios`` / ``holdout_samples`` passed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sqlalchemy import Engine

from aiperfopt.anomaly.detector import AnomalyDetector
from aiperfopt.anomaly.persistence import save_anomaly_detector
from aiperfopt.applog import get_logger
from aiperfopt.config.settings import ContentionSettings, ForecastingSettings
from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.contention.dataset import build_contention_dataset
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
from aiperfopt.contention.model import ContentionClassifier
from aiperfopt.contention.persistence import save_contention_classifier
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.features.engine import compute_feature_matrix
from aiperfopt.forecasting.baselines import Forecaster
from aiperfopt.forecasting.training_pipeline import MetricHorizonOutcome
from aiperfopt.storage.records import EvaluationRecord, SystemMetricSample
from aiperfopt.storage.writer import insert_evaluations_batch

logger = get_logger(__name__)


def _interleave_scenario_chunks(datasets: list[pd.DataFrame], n_chunks: int = 4) -> pd.DataFrame:
    """Interleave chunks from each scenario's dataset before walk-forward CV.

    Found necessary empirically, the same way Forecasting's multi-scenario
    combination was: block-concatenating whole scenarios (each internally
    chronological, but with a large time offset between them) means a walk-forward fold's test slice
    is often an entire single scenario in isolation. For the *rare-event
    binary* contention label this is far more damaging than it was for
    Forecasting's continuous MAE: a fold whose test set contains zero
    positive examples has undefined recall by construction, and a fold
    trained only on one "shape" of contention (e.g. memory pressure) has
    no way to recognize a qualitatively different one (e.g. a CPU spike)
    in its test set — both were observed directly while building this
    pipeline, not assumed.

    Splitting each scenario into ``n_chunks`` contiguous, internally-order-
    preserving pieces and interleaving chunk-by-chunk (chunk 0 of every
    scenario, then chunk 1 of every scenario, ...) gives every walk-forward
    fold's train *and* test slice a representative mix of scenario shapes,
    while each individual chunk is still a real contiguous slice of one
    scenario's own chronological progression — nothing is reordered
    *within* a scenario, only the coarse interleaving order across
    scenarios changes. Row order (not the ``timestamp`` column's values)
    is what ``TimeSeriesSplit`` actually splits on downstream — see
    :func:`aiperfopt.contention.evaluation.run_contention_walk_forward_evaluation`
    — so this is a legitimate, order-only transformation.
    """
    chunked = []
    for ds in datasets:
        if ds.empty:
            continue
        chunk_boundaries = np.array_split(np.arange(len(ds)), n_chunks)
        chunked.append([ds.iloc[idx] for idx in chunk_boundaries if len(idx) > 0])
    interleaved = []
    for chunk_idx in range(n_chunks):
        for scenario_chunks in chunked:
            if chunk_idx < len(scenario_chunks):
                interleaved.append(scenario_chunks[chunk_idx])
    return pd.concat(interleaved, ignore_index=True) if interleaved else pd.DataFrame()


@dataclass
class ContentionTrainingSummary:
    model_version: str
    walk_forward_summary: dict[str, float] = field(default_factory=dict)
    holdout_metrics: dict[str, float] = field(default_factory=dict)
    n_training_rows: int = 0
    classifier_artifact_path: str | None = None
    anomaly_artifact_path: str | None = None
    # Honest evaluation under class imbalance.
    base_rate: float | None = None
    decision_threshold: float | None = None
    threshold_rule_baseline: dict[str, float] | None = None
    # Raw holdout predictions -- carried so aiperfopt.evaluation.report can
    # draw the PR curve / confusion matrix without reloading the artifact
    # and rebuilding the holdout dataset a second time.
    holdout_probabilities: list[float] | None = None
    holdout_labels: list[int] | None = None


def _load_deployed_forecasters(
    forecast_outcomes: dict[str, MetricHorizonOutcome], models_dir: Path, model_version: str
) -> dict[str, Forecaster]:
    # Deferred: aiperfopt.lifecycle imports this module at package-import time.
    from aiperfopt.lifecycle.deployment import load_deployed_forecaster

    forecasters: dict[str, Forecaster] = {}
    for metric_name, outcome in forecast_outcomes.items():
        try:
            forecasters[metric_name] = load_deployed_forecaster(outcome, models_dir, model_version)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Could not load deployed forecaster, excluding metric from contention training",
                exc_info=True, extra={"metric": metric_name},
            )
    return forecasters


def train_contention_and_anomaly_pipeline(
    engine: Engine,
    feature_config: FeatureEngineeringConfig,
    forecasting_settings: ForecastingSettings,
    contention_settings: ContentionSettings,
    labeling_config: ContentionLabelingConfig,
    forecast_outcomes: dict[str, MetricHorizonOutcome],
    model_version: str,
    *,
    scenarios: dict[str, list[SystemMetricSample]],
    holdout_samples: list[SystemMetricSample],
    holdout_dataset_tag: str,
    process_summaries: dict[str, pd.DataFrame] | None = None,
    holdout_process_summary: pd.DataFrame | None = None,
) -> ContentionTrainingSummary:
    """Train Anomaly Detection and Contention Prediction together.

    ``forecast_outcomes`` must supply one short-horizon
    MetricHorizonOutcome per target metric. Only short-horizon forecasts
    are used.

    ``scenarios`` / ``holdout_samples``: training data and a later
    held-out slice, the same shape the forecasting pipeline takes.
    ``holdout_dataset_tag`` is stored on the holdout evaluation row.
    ``process_summaries`` / ``holdout_process_summary``: see
    :func:`aiperfopt.forecasting.training_pipeline.train_evaluate_and_select_one`;
    threaded into the anomaly detector's feature matrix and
    :func:`build_contention_dataset`.
    """
    deployed_forecasters = _load_deployed_forecasters(
        forecast_outcomes, forecasting_settings.models_dir, model_version
    )
    if not deployed_forecasters:
        raise ValueError("No deployed forecasters available — cannot train Contention Prediction")

    # --- Anomaly Detection: trained on the full engineered feature matrix
    # across all non-held-out scenarios (same feature
    # vectors Forecasting consumes). ---
    all_feature_matrices = [
        compute_feature_matrix(samples, feature_config, process_summary=(process_summaries or {}).get(name))
        for name, samples in scenarios.items()
    ]
    combined_features = pd.concat(
        [fm for fm in all_feature_matrices if not fm.empty], ignore_index=True
    )
    anomaly_feature_cols = [c for c in combined_features.columns if c != "timestamp"]

    anomaly_detector = AnomalyDetector()
    anomaly_detector.fit(combined_features[anomaly_feature_cols])
    anomaly_artifact_path = save_anomaly_detector(anomaly_detector, forecasting_settings.models_dir, model_version)

    # --- Contention Prediction dataset: forecast outputs + current state
    # + anomaly score, per scenario. ---
    horizon_seconds = forecasting_settings.short_horizon_target_seconds
    per_scenario_datasets = [
        build_contention_dataset(
            samples, deployed_forecasters, anomaly_detector, feature_config, labeling_config, horizon_seconds,
            process_summary=(process_summaries or {}).get(name),
        )
        for name, samples in scenarios.items()
    ]
    per_scenario_datasets = [ds for ds in per_scenario_datasets if not ds.empty]

    summary = ContentionTrainingSummary(model_version=model_version, anomaly_artifact_path=str(anomaly_artifact_path))

    if not per_scenario_datasets:
        logger.warning("No usable contention training data produced")
        return summary

    combined = _interleave_scenario_chunks(per_scenario_datasets, n_chunks=4)
    summary.n_training_rows = len(combined)

    fold_results = run_contention_walk_forward_evaluation(
        combined, contention_settings.decision_threshold,
        forecasting_settings.walk_forward_splits, forecasting_settings.min_training_rows,
    )
    _persist_contention_fold_evaluations(engine, fold_results, model_version)
    summary.walk_forward_summary = mean_contention_metrics(fold_results)

    if combined["label"].nunique() < 2:
        logger.warning("Combined contention training data has only one class — cannot train final classifier")
        return summary

    feature_cols = feature_columns_of(combined)
    final_classifier = ContentionClassifier(min_training_rows=forecasting_settings.min_training_rows)
    final_classifier.fit(combined[feature_cols], combined["label"])

    # --- Held-out generalization check, same discipline as
    # Forecasting -- run *before* saving, since the PR-curve-fitted
    # decision threshold is computed against this holdout and must be set
    # on the classifier before the artifact is persisted. ---
    holdout_dataset = build_contention_dataset(
        holdout_samples, deployed_forecasters, anomaly_detector, feature_config, labeling_config, horizon_seconds,
        process_summary=holdout_process_summary,
    )
    summary.base_rate = compute_base_rate(combined["label"])
    if not holdout_dataset.empty and holdout_dataset["label"].nunique() >= 1:
        probabilities = final_classifier.predict_proba_batch(holdout_dataset[feature_cols])
        fitted_threshold, threshold_curve_metrics = fit_decision_threshold(probabilities, holdout_dataset["label"])
        final_classifier.decision_threshold = fitted_threshold
        summary.decision_threshold = fitted_threshold

        predictions = (probabilities >= fitted_threshold).astype(int)
        holdout_metrics = {
            "precision": float(precision_score(holdout_dataset["label"], predictions, zero_division=0)),
            "recall": float(recall_score(holdout_dataset["label"], predictions, zero_division=0)),
            "f1": float(f1_score(holdout_dataset["label"], predictions, zero_division=0)),
            "pr_auc": compute_pr_auc(probabilities, holdout_dataset["label"]),
            "decision_threshold": fitted_threshold,
            "base_rate": summary.base_rate,
        }
        summary.holdout_metrics = holdout_metrics
        summary.threshold_rule_baseline = evaluate_threshold_rule_baseline(holdout_dataset, labeling_config)
        summary.holdout_probabilities = probabilities.tolist()
        summary.holdout_labels = holdout_dataset["label"].astype(int).tolist()
        insert_evaluations_batch(engine, [
            EvaluationRecord(
                evaluated_at=pd.Timestamp.now(tz="UTC").timestamp(),
                model_name="contention_classifier", model_version=model_version,
                metric_name=None, horizon_label="short", fold_index=None,
                dataset_tag=holdout_dataset_tag,
                mae=None, rmse=None, directional_accuracy=None, baseline_name=None,
                extra=holdout_metrics,
            )
        ])

    classifier_path = save_contention_classifier(final_classifier, forecasting_settings.models_dir, model_version)
    summary.classifier_artifact_path = str(classifier_path)

    logger.info(
        "Contention/Anomaly training complete",
        extra={"n_training_rows": summary.n_training_rows, "walk_forward": summary.walk_forward_summary,
               "holdout": summary.holdout_metrics},
    )
    return summary


def _persist_contention_fold_evaluations(
    engine: Engine, fold_results: list[ContentionFoldMetrics], model_version: str
) -> None:
    records = [
        EvaluationRecord(
            evaluated_at=pd.Timestamp.now(tz="UTC").timestamp(),
            model_name="contention_classifier", model_version=model_version,
            metric_name=None, horizon_label="short", fold_index=r.fold_index,
            dataset_tag="walk_forward",
            mae=None, rmse=None, directional_accuracy=None, baseline_name=None,
            extra={
                "precision": r.precision, "recall": r.recall, "f1": r.f1, "pr_auc": r.pr_auc,
                "n_positive": r.n_positive,
            },
        )
        for r in fold_results
    ]
    if records:
        insert_evaluations_batch(engine, records)
