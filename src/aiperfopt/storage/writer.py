"""
Write API for the Data Storage subsystem.

Every insert goes through here rather than raw SQL elsewhere in the
codebase, keeping Data Storage as the single place that knows how a
record type maps onto its table (FR-12.4's "only access path" principle
applied internally: no other subsystem should import
:mod:`aiperfopt.storage.schema` directly to build its own INSERT).

Batch functions accept a list and issue one executemany-style statement,
which is what Data Ingestion (Phase 2, FR-2.1: "write to storage in
batches") will call. Row order in the input list is preserved in the
statement sent to SQLite, though retrieval ordering is always determined
by the reader module's explicit ``ORDER BY timestamp`` — chronological
correctness therefore never depends on physical insertion/storage order.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

from sqlalchemy import Engine
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
    raw_process_metrics,
    raw_system_metrics,
    raw_system_metrics_rollup,
    recommendations,
)

logger = get_logger(__name__)


def _row_dict(record) -> dict:  # type: ignore[no-untyped-def]
    """Convert a record dataclass to an insert-ready dict, dropping ``id``.

    ``id`` is omitted so SQLite's autoincrement assigns it; callers never
    supply a pre-existing id when inserting.
    """
    data = asdict(record)
    data.pop("id", None)
    return data


def _insert_one(engine: Engine, table, record) -> int:  # type: ignore[no-untyped-def]
    try:
        with engine.begin() as connection:
            result = connection.execute(table.insert().values(**_row_dict(record)))
            return int(result.inserted_primary_key[0])
    except SQLAlchemyError as exc:
        raise StorageError(
            f"Insert into {table.name} failed", context={"table": table.name}
        ) from exc


def _insert_many(engine: Engine, table, records: Sequence) -> None:  # type: ignore[no-untyped-def]
    if not records:
        return
    try:
        with engine.begin() as connection:
            connection.execute(table.insert(), [_row_dict(r) for r in records])
    except SQLAlchemyError as exc:
        raise StorageError(
            f"Batch insert into {table.name} failed",
            context={"table": table.name, "count": len(records)},
        ) from exc


# --- Raw metric samples -----------------------------------------------------


def insert_system_metric_sample(engine: Engine, sample: SystemMetricSample) -> int:
    return _insert_one(engine, raw_system_metrics, sample)


def insert_system_metric_samples_batch(
    engine: Engine, samples: Sequence[SystemMetricSample]
) -> None:
    _insert_many(engine, raw_system_metrics, samples)


def insert_process_metric_sample(engine: Engine, sample: ProcessMetricSample) -> int:
    return _insert_one(engine, raw_process_metrics, sample)


def insert_process_metric_samples_batch(
    engine: Engine, samples: Sequence[ProcessMetricSample]
) -> None:
    _insert_many(engine, raw_process_metrics, samples)


def insert_process_event(engine: Engine, event: ProcessEvent) -> int:
    return _insert_one(engine, process_events, event)


def insert_process_events_batch(engine: Engine, events: Sequence[ProcessEvent]) -> None:
    _insert_many(engine, process_events, events)


# --- Rollup ------------------------------------------------------------------


def insert_rollup_bucket(engine: Engine, bucket: RollupBucket) -> int:
    return _insert_one(engine, raw_system_metrics_rollup, bucket)


def insert_rollup_buckets_batch(engine: Engine, buckets: Sequence[RollupBucket]) -> None:
    _insert_many(engine, raw_system_metrics_rollup, buckets)


# --- Features ------------------------------------------------------------------


def insert_feature_vector(engine: Engine, record: FeatureVectorRecord) -> int:
    return _insert_one(engine, features, record)


def insert_feature_vectors_batch(engine: Engine, records: Sequence[FeatureVectorRecord]) -> None:
    _insert_many(engine, features, records)


# --- Predictions ---------------------------------------------------------------


def insert_forecast_prediction(engine: Engine, record: ForecastPredictionRecord) -> int:
    return _insert_one(engine, forecast_predictions, record)


def insert_forecast_predictions_batch(
    engine: Engine, records: Sequence[ForecastPredictionRecord]
) -> None:
    _insert_many(engine, forecast_predictions, records)


def insert_contention_prediction(engine: Engine, record: ContentionPredictionRecord) -> int:
    return _insert_one(engine, contention_predictions, record)


# --- Diagnosis (Phase 6) ----------------------------------------------------------


def insert_diagnosis(engine: Engine, record: DiagnosisRecord) -> int:
    return _insert_one(engine, diagnoses, record)


# --- Recommendations (Phase 7) -----------------------------------------------------


def insert_recommendation(engine: Engine, record: RecommendationRecord) -> int:
    return _insert_one(engine, recommendations, record)


# --- Evaluations -----------------------------------------------------------------


def insert_evaluation(engine: Engine, record: EvaluationRecord) -> int:
    return _insert_one(engine, evaluations, record)


def insert_evaluations_batch(engine: Engine, records: Sequence[EvaluationRecord]) -> None:
    _insert_many(engine, evaluations, records)


# --- Model deployments (Phase 8) ----------------------------------------------------


def insert_model_deployment(engine: Engine, record: ModelDeploymentRecord) -> int:
    return _insert_one(engine, model_deployments, record)


# --- Logs --------------------------------------------------------------------------


def insert_log_event(engine: Engine, event: LogEvent) -> int:
    """Persist a significant pipeline event for later querying (Reporting, Phase 11).

    See the module docstring in :mod:`aiperfopt.storage.schema` for why
    this is a deliberate, explicit call site rather than a
    ``logging.Handler`` mirroring every log line.
    """
    return _insert_one(engine, app_logs, event)
