"""
Contention Prediction + Anomaly Detection training pipeline orchestrator.

Depends on Phase 4's *already-trained* forecasters (FR-6.1: Contention
Prediction consumes forecast outputs, not raw metrics) — callers must
supply the winning short-horizon MetricHorizonOutcome for every target
metric, typically the result of already having run
aiperfopt.forecasting.train_and_evaluate_all.

Reuses Forecasting's exact scenario-generation scheme
(generate_training_scenarios / generate_holdout_scenario) so Contention
Prediction and Anomaly Detection train against the identical synthetic
data Forecasting did — deterministic given the same seeds, so this is a
real guarantee, not a coincidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from aiperfopt.anomaly.detector import AnomalyDetector
from aiperfopt.anomaly.persistence import save_anomaly_detector
from aiperfopt.applog import get_logger
from aiperfopt.config.settings import ContentionSettings, ForecastingSettings
from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.contention.dataset import build_contention_dataset
from aiperfopt.contention.evaluation import (
    ContentionFoldMetrics,
    feature_columns_of,
    mean_contention_metrics,
    run_contention_walk_forward_evaluation,
)
from aiperfopt.contention.model import ContentionClassifier
from aiperfopt.contention.persistence import save_contention_classifier
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.features.engine import compute_feature_matrix
from aiperfopt.forecasting.baselines import Forecaster
from aiperfopt.forecasting.deployment import load_deployed_forecaster
from aiperfopt.forecasting.training_pipeline import (
    MetricHorizonOutcome,
    generate_holdout_scenario,
    generate_training_scenarios,
)
from aiperfopt.storage.records import EvaluationRecord, SystemMetricSample
from aiperfopt.storage.writer import insert_evaluations_batch

logger = get_logger(__name__)


def _interleave_scenario_chunks(datasets: list[pd.DataFrame], n_chunks: int = 4) -> pd.DataFrame:
    """Interleave chunks from each scenario's dataset before walk-forward CV.

    Found necessary empirically, the same way Forecasting's multi-scenario
    combination was: block-concatenating whole scenarios (each internally
    chronological, but with a huge synthetic time offset between them —
    see :mod:`aiperfopt.synthetic`) means a walk-forward fold's test slice
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


def _load_deployed_forecasters(
    forecast_outcomes: dict[str, MetricHorizonOutcome], models_dir: Path, model_version: str
) -> dict[str, Forecaster]:
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
    scenario_duration_seconds: float = 900.0,
    interval_seconds: float = 2.0,
    scenarios: dict[str, list[SystemMetricSample]] | None = None,
    holdout_samples: list[SystemMetricSample] | None = None,
    holdout_dataset_tag: str | None = None,
) -> ContentionTrainingSummary:
    """Train Anomaly Detection and Contention Prediction together. The Phase 5 entrypoint.

    ``forecast_outcomes`` must supply one short-horizon
    MetricHorizonOutcome per target metric — the result of an
    already-completed Phase 4 training run for that metric's short
    horizon. Only short-horizon forecasts are used (matching the
    flagship demo's "within the next 30-60 seconds" framing — see
    PROJECT_HANDOFF.md for why medium-horizon contention is deferred,
    not required, for V1 Phase 5).

    ``scenarios`` / ``holdout_samples``: optional pre-built training data
    (the same shape ``aiperfopt.forecasting.training_pipeline`` accepts),
    bypassing synthetic scenario generation. Model Retraining (Phase 8)
    passes the *same* real-usage ``scenarios``/``holdout_samples`` here
    that it passed to Forecasting's retraining call, so Contention
    Prediction and Anomaly Detection retrain against the identical real
    data Forecasting just did — the same determinism guarantee this
    module's docstring already describes for the synthetic case. Must be
    supplied together, or not at all.
    holdout_dataset_tag:
        Overrides the ``dataset_tag`` stored on the holdout evaluation
        row (defaults to a synthetic-scenario-derived tag that is
        misleading for real-usage data -- same rationale as
        :func:`aiperfopt.forecasting.training_pipeline.train_and_evaluate_all`'s
        identically-named parameter).
    """
    if (scenarios is None) != (holdout_samples is None):
        raise ValueError("scenarios and holdout_samples must be supplied together, or not at all")

    deployed_forecasters = _load_deployed_forecasters(
        forecast_outcomes, forecasting_settings.models_dir, model_version
    )
    if not deployed_forecasters:
        raise ValueError("No deployed forecasters available — cannot train Contention Prediction")

    if scenarios is None:
        scenarios = generate_training_scenarios(scenario_duration_seconds, interval_seconds)
        holdout_samples = generate_holdout_scenario(
            scenario_duration_seconds, interval_seconds, scenario_count=len(scenarios)
        )

    # --- Anomaly Detection: trained on the full engineered feature matrix
    # across all non-held-out scenarios (Section 7.2: same feature
    # vectors Forecasting consumes). ---
    all_feature_matrices = [compute_feature_matrix(samples, feature_config) for samples in scenarios.values()]
    combined_features = pd.concat(
        [fm for fm in all_feature_matrices if not fm.empty], ignore_index=True
    )
    anomaly_feature_cols = [c for c in combined_features.columns if c != "timestamp"]

    anomaly_detector = AnomalyDetector()
    anomaly_detector.fit(combined_features[anomaly_feature_cols])
    anomaly_artifact_path = save_anomaly_detector(anomaly_detector, forecasting_settings.models_dir, model_version)

    # --- Contention Prediction dataset: forecast outputs + current state
    # + anomaly score, per scenario (FR-6.1). ---
    horizon_seconds = forecasting_settings.short_horizon_target_seconds
    per_scenario_datasets = [
        build_contention_dataset(
            samples, deployed_forecasters, anomaly_detector, feature_config, labeling_config, horizon_seconds
        )
        for samples in scenarios.values()
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
    classifier_path = save_contention_classifier(final_classifier, forecasting_settings.models_dir, model_version)
    summary.classifier_artifact_path = str(classifier_path)

    # --- Held-out generalization check (Section 11), same discipline as Forecasting. ---
    holdout_dataset = build_contention_dataset(
        holdout_samples, deployed_forecasters, anomaly_detector, feature_config, labeling_config, horizon_seconds
    )
    if not holdout_dataset.empty and holdout_dataset["label"].nunique() >= 1:
        probabilities = final_classifier.predict_proba_batch(holdout_dataset[feature_cols])
        predictions = (probabilities >= contention_settings.decision_threshold).astype(int)
        from sklearn.metrics import f1_score, precision_score, recall_score

        holdout_metrics = {
            "precision": float(precision_score(holdout_dataset["label"], predictions, zero_division=0)),
            "recall": float(recall_score(holdout_dataset["label"], predictions, zero_division=0)),
            "f1": float(f1_score(holdout_dataset["label"], predictions, zero_division=0)),
        }
        summary.holdout_metrics = holdout_metrics
        insert_evaluations_batch(engine, [
            EvaluationRecord(
                evaluated_at=pd.Timestamp.now(tz="UTC").timestamp(),
                model_name="contention_classifier", model_version=model_version,
                metric_name=None, horizon_label="short", fold_index=None,
                dataset_tag=holdout_dataset_tag or "holdout_cpu_stress",
                mae=None, rmse=None, directional_accuracy=None, baseline_name=None,
                extra=holdout_metrics,
            )
        ])

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
            extra={"precision": r.precision, "recall": r.recall, "f1": r.f1, "n_positive": r.n_positive},
        )
        for r in fold_results
    ]
    if records:
        insert_evaluations_batch(engine, records)
