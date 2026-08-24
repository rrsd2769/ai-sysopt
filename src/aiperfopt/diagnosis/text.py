"""
Technical and plain-language explanation text (FR-8.1, FR-8.2).

FR-8.1 requires Developer Mode attribution; FR-8.2 requires the same
underlying diagnosis translated into plain language for General Mode.
Both are produced from the *same* resolved
:class:`~aiperfopt.diagnosis.taxonomy.DiagnosisCategory` and the same
underlying contributing-metrics/processes data — General Mode is not a
separately-computed, potentially-inconsistent narrative, it is a
plain-language rendering of the identical diagnosis (FR-8.3's "single
dominant diagnosis category... regardless of which single label is
user-facing").

Section 6's "Operating Systems concept demonstration" requirement is
implemented here specifically: each category's plain-language template
includes one sentence connecting the observed symptom to the underlying
OS mechanism (scheduling pressure, memory paging, I/O queuing, ...),
rather than leaving that connection implicit or only in Developer Mode.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from aiperfopt.diagnosis.explainer import ForecastExplanation
from aiperfopt.diagnosis.taxonomy import DiagnosisCategory
from aiperfopt.storage.records import ContentionPredictionRecord

_CATEGORY_DISPLAY_NAME: dict[DiagnosisCategory, str] = {
    DiagnosisCategory.CPU_SINGLE_PROCESS: "CPU-Bound Single Process",
    DiagnosisCategory.CPU_MULTI_PROCESS: "CPU-Bound Multi-Process Contention",
    DiagnosisCategory.MEMORY_PRESSURE_GRADUAL: "Memory Pressure — Gradual",
    DiagnosisCategory.MEMORY_PRESSURE_SUDDEN: "Memory Pressure — Sudden",
    DiagnosisCategory.DISK_IO_BOTTLENECK: "Disk I/O Bottleneck",
    DiagnosisCategory.NETWORK_SATURATION: "Network Saturation",
    DiagnosisCategory.CROSS_RESOURCE_CONTENTION: "Cross-Resource Contention",
    DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY: "Unusual Pattern (Anomaly-Flagged)",
}


def _confidence_bucket(probability: float) -> str:
    """Qualitative confidence wording for General Mode.

    A display nicety, not a spec-mandated scale — Developer Mode always
    shows the raw numeric probability alongside this (FR-8.1/FR-8.2 both
    require confidence, but General Mode's audience is better served by
    words than a bare float, per Section 4's "minimal jargon" framing).
    Boundaries are set relative to typical decision-threshold territory
    (Section 11's conservative tuning means a "triggered" prediction is
    already at least moderately confident by construction).
    """
    if probability >= 0.9:
        return "high"
    if probability >= 0.75:
        return "moderate"
    return "some"


def _process_names(contributing_processes: Sequence[Mapping[str, Any]], limit: int = 3) -> list[str]:
    return [str(p.get("name", "an unknown process")) for p in contributing_processes[:limit]]


def _format_process_list(names: Sequence[str]) -> str:
    if not names:
        return "one or more running applications"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def build_plain_language_explanation(
    category: DiagnosisCategory,
    contention_record: ContentionPredictionRecord,
) -> str:
    """General Mode explanation (FR-8.2): plain language, minimal jargon, one OS-concept sentence."""
    all_names = _process_names(contention_record.contributing_processes or [])
    # CPU_SINGLE_PROCESS's whole claim is that *one* process dominates —
    # naming the runner-up alongside it would contradict the category
    # itself, so only this branch uses the single-name form.
    if category == DiagnosisCategory.CPU_SINGLE_PROCESS:
        process_phrase = all_names[0] if all_names else "One application"
    else:
        process_phrase = _format_process_list(all_names)
    confidence_word = _confidence_bucket(contention_record.probability)

    if category == DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY:
        return (
            "Your system's current activity looks different from the patterns it's learned from "
            "past usage. This isn't a specific problem we can point to yet — it just means "
            "predictions are less certain right now than usual, so no action is being recommended."
        )

    templates: dict[DiagnosisCategory, str] = {
        DiagnosisCategory.CPU_SINGLE_PROCESS: (
            f"{process_phrase} is using a large share of your CPU, and there's {confidence_word} "
            "confidence this will cause a slowdown in the next 30-60 seconds. "
            "When one application claims most of the processor's time, the operating system's "
            "scheduler has less time left to give every other running program, so everything else "
            "starts to feel less responsive."
        ),
        DiagnosisCategory.CPU_MULTI_PROCESS: (
            f"Several applications — including {process_phrase} — are competing for CPU time at "
            f"once, with {confidence_word} confidence this leads to a slowdown soon. "
            "The scheduler has to divide processor time across all of them, so as more programs "
            "compete simultaneously, each one gets a smaller share and the whole system slows down."
        ),
        DiagnosisCategory.MEMORY_PRESSURE_GRADUAL: (
            f"Available memory has been steadily shrinking, with {confidence_word} confidence it "
            "will reach a tight point soon. "
            "As free memory runs low, macOS starts moving less-used data out to disk to make room "
            "— and disk is far slower than memory, so things gradually feel more sluggish."
        ),
        DiagnosisCategory.MEMORY_PRESSURE_SUDDEN: (
            f"Memory usage just jumped sharply and is staying high, with {confidence_word} "
            "confidence this causes a slowdown soon. "
            "A sudden, large memory request can force macOS to swap data to disk right away rather "
            "than gradually, which tends to produce a more noticeable stutter than a slow buildup does."
        ),
        DiagnosisCategory.DISK_IO_BOTTLENECK: (
            f"Disk activity is heavy enough that it's likely to bottleneck other work, with "
            f"{confidence_word} confidence. "
            "When one process is reading or writing a lot of data, other processes waiting on the "
            "disk have to queue behind it, which delays anything else that also needs to touch disk."
        ),
        DiagnosisCategory.NETWORK_SATURATION: (
            f"Network activity is approaching a level that could saturate your connection, with "
            f"{confidence_word} confidence. "
            "Once network capacity is close to full, new requests have to wait their turn, so "
            "anything relying on the network — web pages, downloads, video calls — can start to lag."
        ),
        DiagnosisCategory.CROSS_RESOURCE_CONTENTION: (
            f"Several different resources — not just one — appear to be under pressure at the same "
            f"time, with {confidence_word} confidence this leads to a slowdown soon. "
            "When CPU, memory, disk, or network pressure overlap, their effects compound: for "
            "example, a process waiting on disk still occupies memory and scheduler attention, so "
            "combined pressure tends to feel worse than any single resource being stressed alone."
        ),
    }
    return templates[category]


def _forecast_technical_lines(forecast_explanations: Sequence[ForecastExplanation]) -> list[str]:
    lines: list[str] = []
    for fx in forecast_explanations:
        if fx.tier == "not_applicable":
            lines.append(f"- {fx.metric_name}: {fx.note}")
            continue
        top = ", ".join(f"{f.feature_name}={f.value:+.3f}" for f in fx.top_features)
        lines.append(f"- {fx.metric_name} ({fx.forecaster_kind}, {fx.tier} tier): {top or 'no features returned'}")
    return lines


def build_technical_explanation(
    category: DiagnosisCategory,
    contention_record: ContentionPredictionRecord,
    forecast_explanations: Sequence[ForecastExplanation],
    explanation_tier: str,
    trigger_reason: str,
) -> str:
    """Developer Mode explanation (FR-8.1): category, numeric confidence, and full attribution."""
    display_name = _CATEGORY_DISPLAY_NAME[category]
    metrics_line = ", ".join(
        f"{m.get('metric')}={m.get('contribution'):+.3f}" for m in (contention_record.contributing_metrics or [])
    ) or "none"
    processes_line = ", ".join(
        f"{p.get('name')} (pid={p.get('pid')}, value={p.get('value'):.2f})"
        for p in (contention_record.contributing_processes or [])
    ) or "none"

    parts = [
        f"Diagnosis: {display_name}",
        f"Contention probability: {contention_record.probability:.3f} "
        f"(threshold={contention_record.threshold_used:.3f}, triggered={contention_record.triggered})",
        f"Anomaly score: {(contention_record.anomaly_score or 0.0):.3f}",
        f"Explanation tier: {explanation_tier} (trigger: {trigger_reason})",
        f"Contributing metrics (contention classifier coefficients, SHAP-equivalent — "
        f"see aiperfopt.diagnosis.explainer): {metrics_line}",
        f"Contributing processes: {processes_line}",
    ]
    forecast_lines = _forecast_technical_lines(forecast_explanations)
    if forecast_lines:
        parts.append("Per-metric forecast attribution:")
        parts.extend(forecast_lines)
    return "\n".join(parts)
