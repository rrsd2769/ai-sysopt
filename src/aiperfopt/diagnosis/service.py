"""
Diagnosis / Explainable AI orchestrator — the Phase 6 entrypoint (Section 7.1
item 8, FR-8.1-8.3).

:func:`diagnose_contention_event` is the single place that reconciles
Contention Prediction's output with Forecasting's per-metric models into
one persisted :class:`~aiperfopt.storage.records.DiagnosisRecord` — the
"genuine integration point" Section 7.3 describes. It deliberately does
*not* take a live :class:`~aiperfopt.contention.model.ContentionClassifier`
as an argument: everything Diagnosis needs from Contention Prediction is
already present on the triggering
:class:`~aiperfopt.storage.records.ContentionPredictionRecord`
(``contributing_metrics`` already *is* the SHAP-linear-equivalent
attribution — see :mod:`aiperfopt.diagnosis.explainer`'s module
docstring), which keeps Diagnosis a true downstream consumer of
Contention Prediction's *output* rather than reaching back into its
internals — the same boundary FR-6.1 already draws between Contention
Prediction and raw metrics.

What Diagnosis *does* need beyond the contention record:

- The full engineered feature vector at (or immediately before) the
  triggering timestamp — for two things the compact contention-dataset
  row (``forecast_{metric}``/``current_{metric}``/``anomaly_score``
  columns only) doesn't carry: the ``mem_percent_rate_of_change`` feature
  used to distinguish Gradual vs. Sudden memory pressure (taxonomy.py),
  and the full feature set every ``MetricForecaster`` was actually
  trained on, needed for SHAP attribution (explainer.py).
- The currently-deployed :class:`~aiperfopt.forecasting.baselines.Forecaster`
  per metric (LightGBM or a baseline, per Section 10.7's contingency
  plan) — the same ``deployed_forecasters`` mapping
  :func:`aiperfopt.contention.dataset.build_contention_dataset` and
  :func:`aiperfopt.contention.prediction.predict_contention`'s caller
  already assemble via :func:`aiperfopt.forecasting.deployment.load_deployed_forecaster`.

Scope: only the events worth explaining
-------------------------------------------
Consistent with :mod:`aiperfopt.diagnosis.taxonomy`'s resolution-order
documentation, this function returns ``None`` (and persists nothing)
whenever :func:`~aiperfopt.diagnosis.taxonomy.resolve_diagnosis_category`
does — i.e. contention wasn't triggered and there is no significant
anomaly. The overwhelming majority of monitoring cycles are exactly this
"system is healthy" case; manufacturing a diagnosis record for each one
would violate NFR-1.1's overhead budget for no interpretive benefit.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine

from aiperfopt.config.settings import ContentionSettings, DiagnosisSettings
from aiperfopt.diagnosis.explainer import explain_forecast, should_use_shap_tier
from aiperfopt.diagnosis.taxonomy import (
    DiagnosisCategory,
    is_actionable,
    resolve_diagnosis_category,
)
from aiperfopt.diagnosis.text import build_plain_language_explanation, build_technical_explanation
from aiperfopt.forecasting.baselines import Forecaster
from aiperfopt.storage.records import ContentionPredictionRecord, DiagnosisRecord
from aiperfopt.storage.writer import insert_diagnosis

# Which coarse resource type each single-resource category belongs to —
# used only to decide which deployed forecaster(s) are worth explaining
# for a given diagnosis, so Developer Mode isn't handed eight metrics'
# worth of attribution when only one or two are actually relevant to the
# resolved category. Cross-Resource Contention and the anomaly category
# map to no single type: the former explains every deployed metric (no
# single one is more relevant than another, by definition of that
# category), the latter explains none (an anomaly score is a
# feature-vector-distance signal, not a per-metric forecast question).
_CATEGORY_METRIC_NAMES: dict[DiagnosisCategory, tuple[str, ...]] = {
    DiagnosisCategory.CPU_SINGLE_PROCESS: ("cpu_percent", "load_avg_1"),
    DiagnosisCategory.CPU_MULTI_PROCESS: ("cpu_percent", "load_avg_1"),
    DiagnosisCategory.MEMORY_PRESSURE_GRADUAL: ("mem_percent", "swap_percent"),
    DiagnosisCategory.MEMORY_PRESSURE_SUDDEN: ("mem_percent", "swap_percent"),
    DiagnosisCategory.DISK_IO_BOTTLENECK: ("disk_read_bytes", "disk_write_bytes"),
    DiagnosisCategory.NETWORK_SATURATION: ("net_bytes_sent", "net_bytes_recv"),
}


def _metrics_to_explain(category: DiagnosisCategory, deployed_forecasters: dict[str, Forecaster]) -> list[str]:
    if category == DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY:
        return []
    if category == DiagnosisCategory.CROSS_RESOURCE_CONTENTION:
        return list(deployed_forecasters.keys())
    relevant = _CATEGORY_METRIC_NAMES.get(category, ())
    return [name for name in relevant if name in deployed_forecasters]


def diagnose_contention_event(
    engine: Engine,
    contention_record: ContentionPredictionRecord,
    feature_row: pd.Series,
    deployed_forecasters: dict[str, Forecaster],
    contention_settings: ContentionSettings,
    diagnosis_settings: DiagnosisSettings,
    model_version: str,
    explicit_request: bool = False,
) -> DiagnosisRecord | None:
    """Resolve, explain, persist, and return one diagnosis for a contention event.

    Returns ``None`` (persisting nothing) if there is nothing to diagnose
    this cycle — see the module docstring's "Scope" section.
    """
    memory_rate_of_change = feature_row.get("mem_percent_rate_of_change")
    if memory_rate_of_change is not None and pd.isna(memory_rate_of_change):
        memory_rate_of_change = None

    category = resolve_diagnosis_category(
        contention_record,
        memory_rate_of_change,
        contention_settings.anomaly_significant_threshold,
        diagnosis_settings,
    )
    if category is None:
        return None

    use_shap, trigger_reason = should_use_shap_tier(
        contention_record.triggered,
        contention_record.anomaly_score or 0.0,
        contention_settings.anomaly_significant_threshold,
        explicit_request,
    )

    forecast_explanations = [
        explain_forecast(metric_name, deployed_forecasters[metric_name], feature_row, use_shap, diagnosis_settings.shap_top_n)
        for metric_name in _metrics_to_explain(category, deployed_forecasters)
    ]

    explanation_tier = "shap" if use_shap else "lightweight"
    technical_explanation = build_technical_explanation(
        category, contention_record, forecast_explanations, explanation_tier, trigger_reason,
    )
    plain_language_explanation = build_plain_language_explanation(category, contention_record)

    confidence = (
        (contention_record.anomaly_score or 0.0)
        if category == DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY
        else contention_record.probability
    )

    record = DiagnosisRecord(
        generated_at=contention_record.generated_at,
        category=category.value,
        is_actionable=is_actionable(category),
        confidence=confidence,
        explanation_tier=explanation_tier,
        trigger_reason=trigger_reason,
        technical_explanation=technical_explanation,
        plain_language_explanation=plain_language_explanation,
        model_version=model_version,
        forecast_explanations=[fx.to_payload() for fx in forecast_explanations],
        contributing_metrics=contention_record.contributing_metrics,
        contributing_processes=contention_record.contributing_processes,
        anomaly_score=contention_record.anomaly_score,
    )
    diagnosis_id = insert_diagnosis(engine, record)
    record.id = diagnosis_id
    return record
