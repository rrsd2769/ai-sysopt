"""
Diagnosis taxonomy resolution.

Reconciles Contention Prediction's output (probability, coefficient-based
contributing metrics/processes) and Anomaly Detection's score into
exactly one of the eight standardized categories. This is the
genuine integration point: Diagnosis
depends on both Forecasting and Contention Prediction outputs and must
reconcile them into one consistent causal narrative. It is
deliberately rule-based, not a second learned model, in line with the
simpler-mechanism-preferred principle: the
taxonomy is a small, spec-fixed set, not something that
benefits from training a classifier over a classifier's own output.

Resolution order
-----------------
1. **Unusual Pattern (Anomaly-Flagged)** wins outright whenever the
   anomaly score is significant, *regardless* of whether contention was
   also triggered. This matches the rule that anomaly-flagged events bypass
   the Recommendation Engine entirely, and the rationale already applied
   to probability dampening in :mod:`aiperfopt.contention.prediction`: a
   significant anomaly means the classifier is extrapolating well outside
   its training distribution, so any *causal* narrative built from its
   coefficients is untrustworthy even if the (already-dampened)
   probability still cleared the decision threshold.
2. If contention was not triggered and there is no significant anomaly,
   there is nothing to diagnose — :func:`resolve_diagnosis_category`
   returns ``None``. "Every prediction event resolves to exactly
   one category" is read here as scoped to actual triggered/flagged
   events (the flagship demo's "an estimated high probability of
   contention" moment), not to routine below-threshold cycles
   that make up the overwhelming majority of runtime — assigning a
   taxonomy label to "everything is fine" would be manufacturing a
   narrative the explanation requirement was never meant to force.
3. Otherwise, the dominant *resource type* (cpu/memory/disk/network) is
   found by summing |contribution| per type across
   :attr:`ContentionPredictionRecord.contributing_metrics`. If no type
   clearly dominates (its share of the total is below
   ``DiagnosisSettings.cross_resource_dominance_ratio``), the result is
   **Cross-Resource Contention** — the category for exactly this
   "genuinely multi-factor" case, and the taxonomy's own designated
   escape hatch for any resource type this module
   doesn't recognize.
4. A CPU-dominant event is further split into **CPU-Bound Single
   Process** vs. **CPU-Bound Multi-Process Contention** using
   :attr:`ContentionPredictionRecord.contributing_processes`' relative
   dominance. A memory-dominant event is split into **Gradual** vs.
   **Sudden** using the engineered ``mem_percent_rate_of_change`` feature
   (not available from the contention record alone — see
   :mod:`aiperfopt.insight.diagnosis`). Disk and network map directly to
   their single categories (no further split is defined).

Why the anomaly-significance threshold isn't duplicated here
-----------------------------------------------------------------
:class:`~aiperfopt.config.settings.ContentionSettings.anomaly_significant_threshold`
is the single source of truth for "what counts as a significant anomaly" — it already gates
:func:`aiperfopt.anomaly.notifications.emit_anomaly_notification_if_significant`.
Giving :class:`~aiperfopt.config.settings.DiagnosisSettings` its own copy
of that number would be exactly the kind of config-drift bug class this
codebase has otherwise been careful to avoid (see
:func:`aiperfopt.common.timeutils.epoch_seconds_to_utc_datetime` for the
precedent of centralizing a value rather than letting two modules each
hold their own copy) — a future operator could tune one without realizing
the other exists, and get a "significant" anomaly notification with no
matching "Unusual Pattern" diagnosis, or vice versa. Callers pass the one
canonical threshold in explicitly.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Sequence

from aiperfopt.config.settings import DiagnosisSettings
from aiperfopt.storage.records import ContentionPredictionRecord


class DiagnosisCategory(str, Enum):
    """The eight standardized diagnosis categories."""

    CPU_SINGLE_PROCESS = "cpu_bound_single_process"
    CPU_MULTI_PROCESS = "cpu_bound_multi_process_contention"
    MEMORY_PRESSURE_GRADUAL = "memory_pressure_gradual"
    MEMORY_PRESSURE_SUDDEN = "memory_pressure_sudden"
    DISK_IO_BOTTLENECK = "disk_io_bottleneck"
    NETWORK_SATURATION = "network_saturation"
    CROSS_RESOURCE_CONTENTION = "cross_resource_contention"
    UNUSUAL_PATTERN_ANOMALY = "unusual_pattern_anomaly_flagged"


# The one category that never reaches the Recommendation Engine.
# Kept as a set (not a boolean-on-the-enum) so the
# is_actionable rule lives in exactly one place a future Recommendation
# Engine can import directly rather than re-deriving it.
NON_ACTIONABLE_CATEGORIES = frozenset({DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY})


def is_actionable(category: DiagnosisCategory) -> bool:
    return category not in NON_ACTIONABLE_CATEGORIES


# Maps a raw metric column name (after stripping the contention dataset's
# "forecast_"/"current_" prefix — see
# aiperfopt.contention.prediction._strip_feature_prefix) to the coarse
# resource type it represents. Mirrors the core
# metric set; any metric not listed here (there shouldn't be one, given
# aiperfopt.features.config.FeatureEngineeringConfig's
# gauge/counter lists are the exhaustive V1 metric set) is simply excluded
# from resource-type totals rather than raising, keeping this module
# forward-compatible with a metric set change without a hard failure.
_RESOURCE_TYPE_BY_METRIC: dict[str, str] = {
    "cpu_percent": "cpu",
    "load_avg_1": "cpu",
    "mem_percent": "memory",
    "swap_percent": "memory",
    "disk_read_bytes": "disk",
    "disk_write_bytes": "disk",
    "net_bytes_sent": "network",
    "net_bytes_recv": "network",
}

_SINGLE_RESOURCE_CATEGORY: dict[str, DiagnosisCategory] = {
    "disk": DiagnosisCategory.DISK_IO_BOTTLENECK,
    "network": DiagnosisCategory.NETWORK_SATURATION,
}


def _strip_feature_prefix(feature_name: str) -> str:
    """Same stripping rule as aiperfopt.contention.prediction._strip_feature_prefix.

    Duplicated rather than imported: importing from ``contention.prediction``
    into ``diagnosis.taxonomy`` would create a reverse dependency (Contention
    Prediction is upstream of Diagnosis in the data flow, and
    Diagnosis is already the consumer, never the other
    way around). The rule itself is a one-line, spec-independent string
    operation with no risk of drifting out of sync with its counterpart.
    """
    for prefix in ("forecast_", "current_"):
        if feature_name.startswith(prefix):
            return feature_name[len(prefix):]
    return feature_name


def _resource_type_totals(contributing_metrics: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for entry in contributing_metrics:
        metric_name = entry.get("metric")
        contribution = entry.get("contribution")
        if metric_name is None or contribution is None:
            continue
        base_name = _strip_feature_prefix(str(metric_name))
        resource_type = _RESOURCE_TYPE_BY_METRIC.get(base_name)
        if resource_type is None:
            continue  # e.g. "anomaly_score" itself, or an unrecognized column
        totals[resource_type] = totals.get(resource_type, 0.0) + abs(float(contribution))
    return totals


def _resolve_cpu_subcategory(
    contributing_processes: Sequence[Mapping[str, Any]], config: DiagnosisSettings
) -> DiagnosisCategory:
    """Single runaway process vs. several processes contending together.

    Conservative default when process attribution is unavailable or all
    processes measured zero activity: Multi-Process Contention. Claiming
    "one process is responsible" without evidence one actually dominates
    would overclaim a more specific narrative than the data supports —
    the multi-process category is the honest fallback, not the sharper
    one.
    """
    values = [float(p.get("value", 0.0) or 0.0) for p in contributing_processes]
    values = [v for v in values if v > 0]
    if not values:
        return DiagnosisCategory.CPU_MULTI_PROCESS
    if len(values) == 1:
        return DiagnosisCategory.CPU_SINGLE_PROCESS

    total = sum(values)
    top_share = max(values) / total
    if top_share >= config.process_dominance_ratio:
        return DiagnosisCategory.CPU_SINGLE_PROCESS
    return DiagnosisCategory.CPU_MULTI_PROCESS


def _resolve_memory_subcategory(
    memory_rate_of_change: float | None, config: DiagnosisSettings
) -> DiagnosisCategory:
    """Gradual (default) vs. Sudden, based on the engineered rate-of-change feature.

    ``memory_rate_of_change`` is ``None`` when the live engineered feature
    vector doesn't carry ``mem_percent_rate_of_change`` (e.g. too little
    trailing history for a lag lookup within tolerance). Gradual
    is the safe default in that case: it is the less dramatic
    framing, and claiming "Sudden" without the evidence a real spike
    exists would be exactly the kind of overclaiming this module tries to
    avoid elsewhere (see the CPU-subcategory docstring above).
    """
    if memory_rate_of_change is not None and abs(memory_rate_of_change) >= config.memory_sudden_rate_of_change_threshold:
        return DiagnosisCategory.MEMORY_PRESSURE_SUDDEN
    return DiagnosisCategory.MEMORY_PRESSURE_GRADUAL


def resolve_diagnosis_category(
    contention_record: ContentionPredictionRecord,
    memory_rate_of_change: float | None,
    anomaly_significant_threshold: float,
    config: DiagnosisSettings,
) -> DiagnosisCategory | None:
    """Resolve one of the eight categories, or ``None`` if nothing to diagnose.

    Parameters
    ----------
    contention_record:
        The triggering Contention Prediction output (the probability,
        contributing metrics/processes, and anomaly score).
    memory_rate_of_change:
        The current ``mem_percent_rate_of_change`` engineered feature
        value, if available — only consulted when memory is the dominant
        contending resource. See :mod:`aiperfopt.insight.diagnosis` for
        where this is read from the live feature vector.
    anomaly_significant_threshold:
        ``ContentionSettings.anomaly_significant_threshold`` — see the
        module docstring for why this is a parameter, not a
        ``DiagnosisSettings`` field of its own.

    Anomaly is a fallback, not a short-circuit
    -------------------------------------------------------------
    A high anomaly score used to be checked *first*, so it pre-empted
    every other category even when contention was triggered and a
    dominant resource was obvious — the anomaly score was never wrong
    (given F1, it was effectively always ``1.0`` for real data), so this
    made all 7 actionable categories, the SHAP explainer, and the
    recommendation engine unreachable in practice. Contention/dominance
    resolution now always runs first when ``triggered`` is true, and
    always resolves to *something* (Cross-Resource at worst) — anomaly
    only surfaces ``UNUSUAL_PATTERN_ANOMALY`` in the one case nothing
    else had a signal to resolve from: contention wasn't triggered at
    all. The anomaly score itself remains available on
    ``contention_record.anomaly_score`` regardless of which category
    this resolves to — a flag callers can still see, not a category that
    overrides the diagnosis.
    """
    if contention_record.triggered:
        resource_totals = _resource_type_totals(contention_record.contributing_metrics or [])
        total = sum(resource_totals.values())
        if total <= 0 or not resource_totals:
            # Contention was triggered but nothing in contributing_metrics
            # maps to a recognized resource type (e.g. an empty/malformed
            # payload). Cross-Resource is the taxonomy's own escape hatch
            # — refusing to diagnose here
            # would silently drop a triggered event that must resolve
            # to *something*.
            return DiagnosisCategory.CROSS_RESOURCE_CONTENTION

        dominant_type, dominant_value = max(resource_totals.items(), key=lambda kv: kv[1])
        if (dominant_value / total) < config.cross_resource_dominance_ratio:
            return DiagnosisCategory.CROSS_RESOURCE_CONTENTION

        if dominant_type == "cpu":
            return _resolve_cpu_subcategory(contention_record.contributing_processes or [], config)
        if dominant_type == "memory":
            return _resolve_memory_subcategory(memory_rate_of_change, config)
        return _SINGLE_RESOURCE_CATEGORY.get(dominant_type, DiagnosisCategory.CROSS_RESOURCE_CONTENTION)

    anomaly_score = contention_record.anomaly_score or 0.0
    if anomaly_score >= anomaly_significant_threshold:
        return DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY
    return None
