"""
Typed record types for every table in :mod:`aiperfopt.storage.schema`.

Reader functions return these instead of raw SQLAlchemy ``Row``/mapping
objects, and writer functions accept them instead of loose keyword
arguments — giving every later subsystem (Ingestion, Feature Engineering,
Forecasting, ...) a stable, type-checked contract for what a "system
metric sample" or "forecast prediction" looks like, independent of the
underlying table layout.

Every record has ``id: int | None`` (``None`` before insertion) and a
``from_row`` classmethod used internally by the reader module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class SystemMetricSample:
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
    id: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "SystemMetricSample":
        return cls(**{k: row[k] for k in row.keys() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class ProcessMetricSample:
    timestamp: float
    pid: int
    name: str
    cpu_percent: float
    memory_rss: int | None = None
    memory_vms: int | None = None
    memory_percent: float | None = None
    num_threads: int | None = None
    priority: int | None = None
    status: str | None = None
    create_time: float | None = None
    id: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ProcessMetricSample":
        return cls(**{k: row[k] for k in row.keys() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class ProcessEvent:
    timestamp: float
    pid: int
    name: str
    event_type: str  # "launch" | "terminate"
    id: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ProcessEvent":
        return cls(**{k: row[k] for k in row.keys() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class RollupBucket:
    bucket_start: float
    bucket_seconds: int
    sample_count: int
    aggregates: dict[str, dict[str, float]] = field(default_factory=dict)
    id: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "RollupBucket":
        return cls(**{k: row[k] for k in row.keys() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class FeatureVectorRecord:
    timestamp: float
    computed_at: float
    feature_set_version: str
    feature_values: dict[str, float] = field(default_factory=dict)
    id: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "FeatureVectorRecord":
        return cls(**{k: row[k] for k in row.keys() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class ForecastPredictionRecord:
    generated_at: float
    target_timestamp: float
    metric_name: str
    horizon_label: str
    horizon_seconds: int
    predicted_value: float
    model_version: str
    confidence: float | None = None
    id: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ForecastPredictionRecord":
        return cls(**{k: row[k] for k in row.keys() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class ContentionPredictionRecord:
    generated_at: float
    probability: float
    threshold_used: float
    triggered: bool
    model_version: str
    contributing_metrics: list[dict[str, Any]] | None = None
    contributing_processes: list[dict[str, Any]] | None = None
    anomaly_score: float | None = None
    id: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ContentionPredictionRecord":
        return cls(**{k: row[k] for k in row.keys() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class EvaluationRecord:
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
    id: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "EvaluationRecord":
        return cls(**{k: row[k] for k in row.keys() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class ModelDeploymentRecord:
    decided_at: float
    candidate_version: str
    decision: str
    previous_version: str | None = None
    recommendation: str | None = None
    confirmed_by: str | None = None
    comparison_summary: dict[str, Any] | None = None
    id: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ModelDeploymentRecord":
        return cls(**{k: row[k] for k in row.keys() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class DiagnosisRecord:
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
    id: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "DiagnosisRecord":
        return cls(**{k: row[k] for k in row.keys() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class RecommendationRecord:
    generated_at: float
    diagnosis_id: int
    diagnosis_category: str
    actions: list[dict[str, Any]]
    catalog_version: str
    id: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "RecommendationRecord":
        return cls(**{k: row[k] for k in row.keys() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class LogEvent:
    timestamp: float
    level: str
    logger: str
    message: str
    context: dict[str, Any] | None = None
    id: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "LogEvent":
        return cls(**{k: row[k] for k in row.keys() if k in cls.__dataclass_fields__})
