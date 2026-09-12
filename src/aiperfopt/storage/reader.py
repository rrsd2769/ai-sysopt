"""
Read API for the Data Storage subsystem (FR-3.3).

Every range query filters and orders on an indexed timestamp column
(Section 3's ``ix_*_timestamp`` indexes), so these stay fast as raw
tables grow toward their retention bound.

:func:`get_recent_system_metrics` in particular is on the *live inference*
critical path (Section 7.3: "Feature Engineering depends on Data Storage
even for live inference — a historical window is read on every cycle").
It is kept intentionally simple — one indexed range scan — since it will
be called once per monitoring cycle (NFR-2.1: every 1-2 seconds) and its
latency contributes directly to the end-to-end pipeline latency budget
(NFR-3.1: <= 1 second, sample to prediction).
"""

from __future__ import annotations

import time
from typing import Sequence

from sqlalchemy import Engine, Select, select
from sqlalchemy.exc import SQLAlchemyError

from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import StorageError
from aiperfopt.storage.records import (
    ContentionPredictionRecord,
    DiagnosisRecord,
    EvaluationRecord,
    FeatureVectorRecord,
    ForecastPredictionRecord,
    LogEvent,
    ModelDeploymentRecord,
    ProcessEvent,
    ProcessFeatureSummaryRecord,
    ProcessMetricAggregate,
    ProcessMetricSample,
    RecommendationRecord,
    RollupBucket,
    SystemMetricSample,
)
from aiperfopt.storage.schema import (
    app_logs,
    contention_predictions,
    diagnoses,
    evaluations,
    features,
    forecast_predictions,
    model_deployments,
    process_events,
    process_feature_summaries,
    raw_process_metrics,
    raw_process_metrics_other,
    raw_system_metrics,
    raw_system_metrics_rollup,
    recommendations,
)

logger = get_logger(__name__)


def _execute_range(
    engine: Engine,
    table,  # type: ignore[no-untyped-def]
    timestamp_column,  # type: ignore[no-untyped-def]
    start: float | None,
    end: float | None,
    limit: int | None,
    order_desc: bool,
    extra_filter=None,  # type: ignore[no-untyped-def]
) -> list:
    stmt: Select = select(table)
    if start is not None:
        stmt = stmt.where(timestamp_column >= start)
    if end is not None:
        stmt = stmt.where(timestamp_column <= end)
    if extra_filter is not None:
        stmt = stmt.where(extra_filter)
    stmt = stmt.order_by(timestamp_column.desc() if order_desc else timestamp_column.asc())
    if limit is not None:
        stmt = stmt.limit(limit)

    try:
        with engine.connect() as connection:
            result = connection.execute(stmt)
            return [row._mapping for row in result]
    except SQLAlchemyError as exc:
        raise StorageError(
            f"Range query against {table.name} failed", context={"table": table.name}
        ) from exc


# --- Raw metric samples -----------------------------------------------------


def get_system_metrics_range(
    engine: Engine, start: float | None = None, end: float | None = None, limit: int | None = None
) -> list[SystemMetricSample]:
    rows = _execute_range(
        engine, raw_system_metrics, raw_system_metrics.c.timestamp, start, end, limit, False
    )
    return [SystemMetricSample.from_row(r) for r in rows]


def get_recent_system_metrics(
    engine: Engine, window_seconds: float, as_of: float | None = None
) -> list[SystemMetricSample]:
    """Read the trailing window of system metrics ending at ``as_of`` (default: now).

    This is the query Feature Engineering (Phase 3) will call once per
    monitoring cycle to compute lag/rolling-window features for live
    inference (Section 7.3).
    """
    reference = as_of if as_of is not None else time.time()
    return get_system_metrics_range(engine, start=reference - window_seconds, end=reference)


def get_process_metrics_range(
    engine: Engine,
    start: float | None = None,
    end: float | None = None,
    pid: int | None = None,
    limit: int | None = None,
) -> list[ProcessMetricSample]:
    extra_filter = raw_process_metrics.c.pid == pid if pid is not None else None
    rows = _execute_range(
        engine,
        raw_process_metrics,
        raw_process_metrics.c.timestamp,
        start,
        end,
        limit,
        False,
        extra_filter,
    )
    return [ProcessMetricSample.from_row(r) for r in rows]


def get_process_events_range(
    engine: Engine, start: float | None = None, end: float | None = None, limit: int | None = None
) -> list[ProcessEvent]:
    rows = _execute_range(
        engine, process_events, process_events.c.timestamp, start, end, limit, False
    )
    return [ProcessEvent.from_row(r) for r in rows]


def get_process_metric_aggregates_range(
    engine: Engine, start: float | None = None, end: float | None = None, limit: int | None = None
) -> list[ProcessMetricAggregate]:
    rows = _execute_range(
        engine,
        raw_process_metrics_other,
        raw_process_metrics_other.c.timestamp,
        start,
        end,
        limit,
        False,
    )
    return [ProcessMetricAggregate.from_row(r) for r in rows]


# --- Rollup ------------------------------------------------------------------


def get_rollup_range(
    engine: Engine, start: float | None = None, end: float | None = None, limit: int | None = None
) -> list[RollupBucket]:
    """Read rolled-up (downsampled) history — used for reporting beyond the
    full-resolution retention window (NFR-4.1)."""
    rows = _execute_range(
        engine,
        raw_system_metrics_rollup,
        raw_system_metrics_rollup.c.bucket_start,
        start,
        end,
        limit,
        False,
    )
    return [RollupBucket.from_row(r) for r in rows]


# --- Features ------------------------------------------------------------------


def get_features_range(
    engine: Engine, start: float | None = None, end: float | None = None, limit: int | None = None
) -> list[FeatureVectorRecord]:
    rows = _execute_range(engine, features, features.c.timestamp, start, end, limit, False)
    return [FeatureVectorRecord.from_row(r) for r in rows]


def get_latest_feature_vector(engine: Engine) -> FeatureVectorRecord | None:
    rows = _execute_range(engine, features, features.c.timestamp, None, None, 1, True)
    return FeatureVectorRecord.from_row(rows[0]) if rows else None


def get_process_feature_summaries_range(
    engine: Engine, start: float | None = None, end: float | None = None, limit: int | None = None
) -> list[ProcessFeatureSummaryRecord]:
    rows = _execute_range(
        engine,
        process_feature_summaries,
        process_feature_summaries.c.timestamp,
        start,
        end,
        limit,
        False,
    )
    return [ProcessFeatureSummaryRecord.from_row(r) for r in rows]


# --- Predictions ---------------------------------------------------------------


def get_forecast_predictions_range(
    engine: Engine,
    start: float | None = None,
    end: float | None = None,
    metric_name: str | None = None,
    limit: int | None = None,
) -> list[ForecastPredictionRecord]:
    extra_filter = (
        forecast_predictions.c.metric_name == metric_name if metric_name is not None else None
    )
    rows = _execute_range(
        engine,
        forecast_predictions,
        forecast_predictions.c.generated_at,
        start,
        end,
        limit,
        False,
        extra_filter,
    )
    return [ForecastPredictionRecord.from_row(r) for r in rows]


def get_contention_predictions_range(
    engine: Engine, start: float | None = None, end: float | None = None, limit: int | None = None
) -> list[ContentionPredictionRecord]:
    rows = _execute_range(
        engine,
        contention_predictions,
        contention_predictions.c.generated_at,
        start,
        end,
        limit,
        False,
    )
    return [ContentionPredictionRecord.from_row(r) for r in rows]


# --- Diagnosis (Phase 6) -----------------------------------------------------------


def get_diagnoses_range(
    engine: Engine, start: float | None = None, end: float | None = None, limit: int | None = None
) -> list[DiagnosisRecord]:
    rows = _execute_range(engine, diagnoses, diagnoses.c.generated_at, start, end, limit, False)
    return [DiagnosisRecord.from_row(r) for r in rows]


# --- Recommendations (Phase 7) ------------------------------------------------------


def get_recommendations_range(
    engine: Engine, start: float | None = None, end: float | None = None, limit: int | None = None
) -> list[RecommendationRecord]:
    rows = _execute_range(engine, recommendations, recommendations.c.generated_at, start, end, limit, False)
    return [RecommendationRecord.from_row(r) for r in rows]


# --- Evaluations -----------------------------------------------------------------


def get_evaluations_range(
    engine: Engine, start: float | None = None, end: float | None = None, limit: int | None = None
) -> list[EvaluationRecord]:
    rows = _execute_range(engine, evaluations, evaluations.c.evaluated_at, start, end, limit, False)
    return [EvaluationRecord.from_row(r) for r in rows]


def get_evaluations_by_version(engine: Engine, model_version: str) -> list[EvaluationRecord]:
    """Every evaluation row (walk-forward fold + holdout) recorded for one model_version.

    Used by Retraining (Phase 8) to reconstruct a previously-deployed
    version's holdout results for comparison against a new candidate,
    without needing a separate "which version was this trained as"
    lookup elsewhere.
    """
    rows = _execute_range(
        engine, evaluations, evaluations.c.evaluated_at, None, None, None, False,
        evaluations.c.model_version == model_version,
    )
    return [EvaluationRecord.from_row(r) for r in rows]


# --- Model deployments (Phase 8) ----------------------------------------------------


def get_model_deployments_range(
    engine: Engine, start: float | None = None, end: float | None = None, limit: int | None = None
) -> list[ModelDeploymentRecord]:
    rows = _execute_range(engine, model_deployments, model_deployments.c.decided_at, start, end, limit, False)
    return [ModelDeploymentRecord.from_row(r) for r in rows]


def get_latest_model_deployment(engine: Engine) -> ModelDeploymentRecord | None:
    rows = _execute_range(engine, model_deployments, model_deployments.c.decided_at, None, None, 1, True)
    return ModelDeploymentRecord.from_row(rows[0]) if rows else None


# --- Logs --------------------------------------------------------------------------


def get_log_events_range(
    engine: Engine, start: float | None = None, end: float | None = None, limit: int | None = None
) -> list[LogEvent]:
    rows = _execute_range(engine, app_logs, app_logs.c.timestamp, start, end, limit, False)
    return [LogEvent.from_row(r) for r in rows]
