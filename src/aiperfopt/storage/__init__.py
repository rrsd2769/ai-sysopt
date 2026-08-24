"""Data Storage subsystem (Section 7.1, item 3).

- ``engine``            -> connection bootstrap (Phase 0)
- ``schema``             -> table definitions (Phase 1, FR-3.1)
- ``schema_management``  -> create/verify schema (Phase 1)
- ``records``            -> typed row dataclasses (Phase 1)
- ``writer`` / ``reader`` -> the only supported way to read/write storage
- ``rollup``             -> downsampling + retention (Phase 1, FR-3.2)
"""

from aiperfopt.storage.engine import (
    build_sqlite_url,
    check_connectivity,
    create_storage_engine,
    engine_connection,
)
from aiperfopt.storage.reader import (
    get_contention_predictions_range,
    get_diagnoses_range,
    get_evaluations_by_version,
    get_evaluations_range,
    get_latest_model_deployment,
    get_model_deployments_range,
    get_recommendations_range,
    get_features_range,
    get_forecast_predictions_range,
    get_latest_feature_vector,
    get_log_events_range,
    get_process_events_range,
    get_process_metrics_range,
    get_recent_system_metrics,
    get_rollup_range,
    get_system_metrics_range,
)
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
from aiperfopt.storage.rollup import (
    PruneResult,
    RollupResult,
    apply_retention_policy,
    prune_process_level_data,
    rollup_system_metrics,
)
from aiperfopt.storage.schema_management import create_all_tables, verify_schema
from aiperfopt.storage.writer import (
    insert_contention_prediction,
    insert_diagnosis,
    insert_evaluation,
    insert_evaluations_batch,
    insert_feature_vector,
    insert_feature_vectors_batch,
    insert_forecast_prediction,
    insert_forecast_predictions_batch,
    insert_log_event,
    insert_model_deployment,
    insert_process_event,
    insert_process_events_batch,
    insert_process_metric_sample,
    insert_process_metric_samples_batch,
    insert_recommendation,
    insert_rollup_bucket,
    insert_rollup_buckets_batch,
    insert_system_metric_sample,
    insert_system_metric_samples_batch,
)

__all__ = [
    # engine
    "build_sqlite_url",
    "check_connectivity",
    "create_storage_engine",
    "engine_connection",
    # schema management
    "create_all_tables",
    "verify_schema",
    # records
    "ContentionPredictionRecord",
    "DiagnosisRecord",
    "EvaluationRecord",
    "FeatureVectorRecord",
    "ForecastPredictionRecord",
    "LogEvent",
    "ModelDeploymentRecord",
    "ProcessEvent",
    "ProcessMetricSample",
    "RecommendationRecord",
    "RollupBucket",
    "SystemMetricSample",
    # writer
    "insert_contention_prediction",
    "insert_diagnosis",
    "insert_evaluation",
    "insert_evaluations_batch",
    "insert_feature_vector",
    "insert_feature_vectors_batch",
    "insert_forecast_prediction",
    "insert_forecast_predictions_batch",
    "insert_log_event",
    "insert_model_deployment",
    "insert_process_event",
    "insert_process_events_batch",
    "insert_process_metric_sample",
    "insert_process_metric_samples_batch",
    "insert_recommendation",
    "insert_rollup_bucket",
    "insert_rollup_buckets_batch",
    "insert_system_metric_sample",
    "insert_system_metric_samples_batch",
    # reader
    "get_contention_predictions_range",
    "get_diagnoses_range",
    "get_evaluations_by_version",
    "get_evaluations_range",
    "get_features_range",
    "get_forecast_predictions_range",
    "get_latest_feature_vector",
    "get_latest_model_deployment",
    "get_log_events_range",
    "get_model_deployments_range",
    "get_process_events_range",
    "get_process_metrics_range",
    "get_recent_system_metrics",
    "get_recommendations_range",
    "get_rollup_range",
    "get_system_metrics_range",
    # rollup
    "PruneResult",
    "RollupResult",
    "apply_retention_policy",
    "prune_process_level_data",
    "rollup_system_metrics",
]
