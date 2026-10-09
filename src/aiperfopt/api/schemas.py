"""
Pydantic schemas for the API layer.

Every storage record type (:mod:`aiperfopt.storage.records`) already has
exactly the fields a REST response needs — these wrapper models exist
only so FastAPI can validate them (``from_attributes=True`` lets a
Pydantic model be built directly from a dataclass instance, no manual
field copying) and auto-generate accurate OpenAPI documentation (FR-12.3)
from them. None of these add or rename fields relative to their source
dataclass; that 1:1 mapping is deliberate; a REST consumer reading this
API's docs sees exactly the same shape a Python caller reading
``storage/records.py`` would.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _RecordSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class SystemMetricSampleOut(_RecordSchema):
    id: int | None = None
    timestamp: float
    cpu_percent: float
    mem_percent: float
    cpu_count_logical: int | None = None
    mem_total: int | None = None
    mem_used: int | None = None
    mem_available: int | None = None
    swap_total: int | None = None
    swap_used: int | None = None
    swap_percent: float | None = None
    disk_read_bytes: int | None = None
    disk_write_bytes: int | None = None
    disk_read_count: int | None = None
    disk_write_count: int | None = None
    net_bytes_sent: int | None = None
    net_bytes_recv: int | None = None
    net_packets_sent: int | None = None
    net_packets_recv: int | None = None
    load_avg_1: float | None = None
    load_avg_5: float | None = None
    load_avg_15: float | None = None
    thread_count_total: int | None = None
    battery_percent: float | None = None
    power_plugged: bool | None = None


class FeatureVectorOut(_RecordSchema):
    id: int | None = None
    timestamp: float
    computed_at: float
    feature_set_version: str
    feature_values: dict[str, float | None] = Field(default_factory=dict)


class ForecastPredictionOut(_RecordSchema):
    id: int | None = None
    generated_at: float
    target_timestamp: float
    metric_name: str
    horizon_label: str
    horizon_seconds: int
    predicted_value: float
    model_version: str
    confidence: float | None = None


class ContentionPredictionOut(_RecordSchema):
    id: int | None = None
    generated_at: float
    probability: float
    threshold_used: float
    triggered: bool
    model_version: str
    contributing_metrics: list[dict[str, Any]] | None = None
    contributing_processes: list[dict[str, Any]] | None = None
    anomaly_score: float | None = None


class DiagnosisOut(_RecordSchema):
    id: int | None = None
    generated_at: float
    category: str
    is_actionable: bool
    confidence: float
    explanation_tier: str
    trigger_reason: str
    technical_explanation: str
    plain_language_explanation: str
    model_version: str
    forecast_explanations: list[dict[str, Any]] | None = None
    contributing_metrics: list[dict[str, Any]] | None = None
    contributing_processes: list[dict[str, Any]] | None = None
    anomaly_score: float | None = None


class RecommendationOut(_RecordSchema):
    id: int | None = None
    generated_at: float
    diagnosis_id: int
    diagnosis_category: str
    actions: list[dict[str, Any]]
    catalog_version: str


class EvaluationOut(_RecordSchema):
    id: int | None = None
    evaluated_at: float
    model_name: str
    model_version: str
    dataset_tag: str
    metric_name: str | None = None
    horizon_label: str | None = None
    fold_index: int | None = None
    mae: float | None = None
    rmse: float | None = None
    directional_accuracy: float | None = None
    baseline_name: str | None = None
    extra: dict[str, Any] | None = None


class ModelDeploymentOut(_RecordSchema):
    id: int | None = None
    decided_at: float
    candidate_version: str
    previous_version: str | None = None
    decision: str
    recommendation: str | None = None
    confirmed_by: str | None = None
    comparison_summary: dict[str, Any] | None = None


class RecommendationActionOut(_RecordSchema):
    """One curated action from a :class:`RecommendationOut.actions` entry.

    Not used directly as a response model (``actions`` is a plain list of
    dicts sourced from ``aiperfopt/resources/recommendations.yaml``, per Phase 7's
    design — see that phase's YAML-is-the-source-of-truth rationale) —
    provided here only so the OpenAPI schema documents the shape a
    consumer should expect inside each dict.
    """

    title: str
    description: str
    rationale: str


# --- Live state ----------------------------------------------------------------------


class LiveStateOut(BaseModel):
    """One snapshot from the live-inference loop (FR-12.1).

    Fields are progressively optional in the order the loop's pipeline
    can stop early (NFR-11.1/11.2 graceful degradation): no feature
    vector yet (insufficient trailing history) -> no active model
    deployed yet -> a model deployed but this cycle produced no
    triggered contention event to diagnose. A client should treat
    ``None`` fields as "nothing to show for this stage yet," not an
    error — ``error`` carries anything that genuinely went wrong.
    """

    model_config = ConfigDict(from_attributes=True)

    timestamp: float
    latest_metrics: SystemMetricSampleOut | None = None
    feature_vector: dict[str, float | None] | None = None
    active_model_version: str | None = None
    forecasts: list[ForecastPredictionOut] | None = None
    contention: ContentionPredictionOut | None = None
    diagnosis: DiagnosisOut | None = None
    recommendation: RecommendationOut | None = None
    anomaly_notification: bool = False
    error: str | None = None


# --- Retraining ------------------------------------------------------------------------


class RetrainRequest(BaseModel):
    new_model_version: str = Field(min_length=1)
    target_metrics: list[str] | None = None
    start: float | None = None
    end: float | None = None


class MetricComparisonOut(BaseModel):
    metric_name: str
    horizon_label: str
    candidate_mae: float | None
    baseline_mae: float | None
    candidate_directional_accuracy: float | None
    baseline_directional_accuracy: float | None
    regressed: bool
    note: str | None = None


class ContentionComparisonOut(BaseModel):
    candidate_f1: float | None
    baseline_f1: float | None
    regressed: bool
    note: str | None = None


class RetrainingComparisonOut(BaseModel):
    candidate_version: str
    active_version: str | None
    recommendation: str
    reasons: list[str]
    metric_comparisons: list[MetricComparisonOut]
    contention_comparison: ContentionComparisonOut | None = None


class ForecastGateCriterionOut(BaseModel):
    metric_name: str
    horizon_label: str
    candidate_mae: float | None
    persistence_mae: float | None
    candidate_directional_accuracy: float | None
    passed: bool
    note: str | None = None


class ContentionGateCriterionOut(BaseModel):
    candidate_f1: float | None
    baseline_f1: float | None
    passed: bool
    note: str | None = None


class SkillGateOut(BaseModel):
    """Phase 6's absolute quality bar -- see :func:`aiperfopt.retraining.
    comparison.evaluate_skill_gate`. Unlike ``RetrainingComparisonOut``'s
    ``recommendation`` (advisory), ``passed=False`` here blocks
    ``/retraining/confirm`` even with ``confirmed=True``."""

    passed: bool
    reasons: list[str]
    forecast_criteria: list[ForecastGateCriterionOut]
    contention_criterion: ContentionGateCriterionOut | None = None


class RetrainingCandidateOut(BaseModel):
    model_version: str
    data_source: str
    comparison: RetrainingComparisonOut
    gate: SkillGateOut


class ConfirmDeploymentRequest(BaseModel):
    candidate_version: str = Field(min_length=1)
    comparison: RetrainingComparisonOut
    gate: SkillGateOut
    confirmed: bool
    confirmed_by: str = "human"


class RollbackRequest(BaseModel):
    reason: str = Field(min_length=1)


# --- Reporting ---------------------------------------------------------------------


class PredictedVsActualPointOut(BaseModel):
    """One joined predicted/actual pair for the FR-13.3 comparison view.

    ``actual_value`` (and therefore ``error``) is ``None`` when no
    observed sample fell close enough to ``target_timestamp`` to count
    as a match (see ``api/routers/reporting.py``) -- a client should
    render that as "not yet available", not an error state.
    """

    metric_name: str
    horizon_label: str
    generated_at: float
    target_timestamp: float
    predicted_value: float
    actual_value: float | None
    error: float | None
    confidence: float | None
    model_version: str


# --- Reporting summary (Phase 11, FR-14.1/FR-14.2) --------------------------------


class SystemMetricSummaryOut(BaseModel):
    metric_name: str
    avg: float | None
    minimum: float | None
    maximum: float | None
    sample_count: int


class PredictionAccuracySummaryOut(BaseModel):
    metric_name: str
    horizon_label: str
    mae: float | None
    rmse: float | None
    directional_accuracy: float | None
    matched_sample_count: int
    prediction_count: int


class ContentionSummaryOut(BaseModel):
    prediction_count: int
    triggered_count: int
    avg_probability: float | None


class DiagnosisSummaryOut(BaseModel):
    total_count: int
    actionable_count: int
    avg_confidence: float | None
    counts_by_category: dict[str, int]


class RecommendationSummaryOut(BaseModel):
    total_count: int
    total_actions: int
    counts_by_diagnosis_category: dict[str, int]


class ReportSummaryOut(BaseModel):
    """FR-14.1's exportable summary: system-metrics context plus, per
    FR-14.2, prediction accuracy -- and, per ADL-22's flagship-pillar
    scoping, diagnosis and recommendation summaries alongside it.
    """

    start: float
    end: float
    generated_at: float
    system_metrics: list[SystemMetricSummaryOut]
    prediction_accuracy: list[PredictionAccuracySummaryOut]
    contention: ContentionSummaryOut
    diagnosis: DiagnosisSummaryOut
    recommendation: RecommendationSummaryOut


class ReadinessOut(BaseModel):
    """Why the live pipeline can or cannot predict right now (``GET /readiness``)."""

    state: str
    detail: str
    history_seconds: float
    history_seconds_required: float
    active_model_version: str | None = None
    can_predict: bool
