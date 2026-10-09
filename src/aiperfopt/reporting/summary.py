"""
Builds one ``ReportSummary`` for a time window (FR-14.1's "exportable
daily/weekly summaries"), reusing every already-tested reader/evaluation
function rather than re-deriving any of them:

- **Prediction accuracy** (FR-14.2's explicit requirement -- "tying
  reporting back to the prediction pillar") reuses
  ``reporting.predicted_vs_actual.join_predicted_vs_actual`` for the
  matching and ``forecasting.evaluation.compute_mae_rmse``/
  ``compute_directional_accuracy`` for the same metrics Section 11's
  walk-forward validation reports, computed here over live-observed
  predictions instead of held-out training folds.
- **Diagnosis and recommendation summaries** exist because Section 6's
  other two flagship pillars (explainability, traceable recommendations)
  deserve the same "how did this actually perform" treatment prediction
  gets -- not because ADL-22 wants generic dashboarding of every table.
  System-metrics summaries are the one exception, included only as
  interpretive context for the other three (e.g. "CPU averaged 80% this
  week" explains *why* CPU-Bound diagnoses were frequent), not as an end
  in themselves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean

import numpy as np
from sqlalchemy import Engine

from aiperfopt.insight.taxonomy import DiagnosisCategory
from aiperfopt.forecasting.evaluation import compute_directional_accuracy, compute_mae_rmse
from aiperfopt.reporting.predicted_vs_actual import join_predicted_vs_actual
from aiperfopt.storage.reader import (
    get_contention_predictions_range,
    get_diagnoses_range,
    get_forecast_predictions_range,
    get_recommendations_range,
    get_system_metrics_range,
)

# Section 9's core monitoring metric set, restricted to the numeric
# columns that have a meaningful avg/min/max (excludes booleans like
# power_plugged and identifiers). This is the *system-metrics context*
# default; prediction-accuracy coverage is independent of this list --
# it's driven entirely by whatever metrics actually have predictions
# in the requested window (see build_prediction_accuracy_summary).
DEFAULT_REPORT_METRICS = (
    "cpu_percent", "mem_percent", "swap_percent",
    "disk_read_bytes", "disk_write_bytes",
    "net_bytes_sent", "net_bytes_recv",
    "load_avg_1", "thread_count_total",
)

PERIOD_SECONDS = {"daily": 24 * 3600.0, "weekly": 7 * 24 * 3600.0}


@dataclass(frozen=True)
class SystemMetricSummary:
    metric_name: str
    avg: float | None
    minimum: float | None
    maximum: float | None
    sample_count: int


@dataclass(frozen=True)
class PredictionAccuracySummary:
    metric_name: str
    horizon_label: str
    mae: float | None
    rmse: float | None
    directional_accuracy: float | None
    matched_sample_count: int
    prediction_count: int


@dataclass(frozen=True)
class ContentionSummary:
    prediction_count: int
    triggered_count: int
    avg_probability: float | None


@dataclass(frozen=True)
class DiagnosisSummary:
    total_count: int
    actionable_count: int
    avg_confidence: float | None
    counts_by_category: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RecommendationSummary:
    total_count: int
    total_actions: int
    counts_by_diagnosis_category: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ReportSummary:
    start: float
    end: float
    generated_at: float
    system_metrics: list[SystemMetricSummary]
    prediction_accuracy: list[PredictionAccuracySummary]
    contention: ContentionSummary
    diagnosis: DiagnosisSummary
    recommendation: RecommendationSummary


def resolve_report_window(
    period: str | None, start: float | None, end: float | None
) -> tuple[float, float]:
    """Explicit ``start``/``end`` always wins; otherwise ``period`` ("daily"/"weekly")
    resolves a window ending at ``end`` (default: now) and spanning that period's length."""
    if start is not None and end is not None:
        if start >= end:
            raise ValueError("start must be before end")
        return start, end
    if period not in PERIOD_SECONDS:
        raise ValueError(f"period must be one of {sorted(PERIOD_SECONDS)}, or pass explicit start and end")
    resolved_end = end if end is not None else time.time()
    resolved_start = start if start is not None else resolved_end - PERIOD_SECONDS[period]
    return resolved_start, resolved_end


def build_system_metrics_summary(
    engine: Engine, start: float, end: float, metric_names: tuple[str, ...] = DEFAULT_REPORT_METRICS
) -> list[SystemMetricSummary]:
    samples = get_system_metrics_range(engine, start=start, end=end)
    summaries = []
    for name in metric_names:
        values = [v for s in samples if (v := getattr(s, name, None)) is not None]
        if values:
            summaries.append(SystemMetricSummary(name, mean(values), min(values), max(values), len(values)))
        else:
            summaries.append(SystemMetricSummary(name, None, None, None, 0))
    return summaries


def build_prediction_accuracy_summary(
    engine: Engine, start: float, end: float, stability_threshold: float = 0.02
) -> list[PredictionAccuracySummary]:
    """One entry per (metric_name, horizon_label) that actually had a prediction in the window.

    Unlike the system-metrics summary, this is not driven by a fixed
    metric list -- it reports on whatever forecasting actually produced,
    so a metric with no predictions this window simply doesn't appear
    rather than showing as a row of ``None``s.
    """
    predictions = get_forecast_predictions_range(engine, start=start, end=end)
    combos = sorted({(p.metric_name, p.horizon_label) for p in predictions})

    summaries = []
    for metric_name, horizon_label in combos:
        points = join_predicted_vs_actual(engine, metric_name, start=start, end=end, horizon_label=horizon_label)
        matched = [p for p in points if p.actual_value is not None and p.current_value is not None]

        if matched:
            y_true = np.array([p.actual_value for p in matched])
            y_pred = np.array([p.predicted_value for p in matched])
            current = np.array([p.current_value for p in matched])
            mae, rmse = compute_mae_rmse(y_true, y_pred)
            directional = compute_directional_accuracy(current, y_true, y_pred, stability_threshold)
            directional = None if np.isnan(directional) else directional
        else:
            mae = rmse = directional = None

        summaries.append(
            PredictionAccuracySummary(
                metric_name=metric_name, horizon_label=horizon_label, mae=mae, rmse=rmse,
                directional_accuracy=directional, matched_sample_count=len(matched),
                prediction_count=len(points),
            )
        )
    return summaries


def build_contention_summary(engine: Engine, start: float, end: float) -> ContentionSummary:
    predictions = get_contention_predictions_range(engine, start=start, end=end)
    triggered = sum(1 for p in predictions if p.triggered)
    avg_probability = mean(p.probability for p in predictions) if predictions else None
    return ContentionSummary(len(predictions), triggered, avg_probability)


def build_diagnosis_summary(engine: Engine, start: float, end: float) -> DiagnosisSummary:
    diagnoses = get_diagnoses_range(engine, start=start, end=end)
    # Zero-filled for every real taxonomy category (Section 12) so a
    # report consumer never has to special-case a missing key for a
    # category that simply didn't occur this window.
    counts = {category.value: 0 for category in DiagnosisCategory}
    for d in diagnoses:
        counts[d.category] = counts.get(d.category, 0) + 1
    actionable = sum(1 for d in diagnoses if d.is_actionable)
    avg_confidence = mean(d.confidence for d in diagnoses) if diagnoses else None
    return DiagnosisSummary(len(diagnoses), actionable, avg_confidence, counts)


def build_recommendation_summary(engine: Engine, start: float, end: float) -> RecommendationSummary:
    recommendations = get_recommendations_range(engine, start=start, end=end)
    counts: dict[str, int] = {}
    total_actions = 0
    for r in recommendations:
        counts[r.diagnosis_category] = counts.get(r.diagnosis_category, 0) + 1
        total_actions += len(r.actions)
    return RecommendationSummary(len(recommendations), total_actions, counts)


def build_report(
    engine: Engine,
    period: str | None = None,
    start: float | None = None,
    end: float | None = None,
    system_metrics: tuple[str, ...] = DEFAULT_REPORT_METRICS,
    stability_threshold: float = 0.02,
) -> ReportSummary:
    """The single entry point every caller (the API router, a future export/CLI path) should use."""
    resolved_start, resolved_end = resolve_report_window(period, start, end)
    return ReportSummary(
        start=resolved_start,
        end=resolved_end,
        generated_at=time.time(),
        system_metrics=build_system_metrics_summary(engine, resolved_start, resolved_end, system_metrics),
        prediction_accuracy=build_prediction_accuracy_summary(
            engine, resolved_start, resolved_end, stability_threshold
        ),
        contention=build_contention_summary(engine, resolved_start, resolved_end),
        diagnosis=build_diagnosis_summary(engine, resolved_start, resolved_end),
        recommendation=build_recommendation_summary(engine, resolved_start, resolved_end),
    )
