"""
Predicted-vs-actual join: pairs each forecast with the actual value
nearest its ``target_timestamp`` -- and, for the accuracy
reporting, also the actual value nearest its ``generated_at`` (the
"current value" a directional-accuracy check needs a reference point
for, the same way ``forecasting.evaluation.compute_directional_accuracy``
needs one during walk-forward validation).

Originally built directly inside ``api/routers/reporting.py`` for the
comparison view; moved here once a second consumer (``summary.py``'s prediction-accuracy report)
needed the same join, per that router module's own docstring flagging
this as the intended landing spot.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from sqlalchemy import Engine

from aiperfopt.storage.reader import get_forecast_predictions_range, get_system_metrics_range
from aiperfopt.storage.records import SystemMetricSample

# The set of raw_system_metrics columns that forecasting ever trains a
# model against (aiperfopt.forecasting.dataset raises KeyError on
# anything else) -- rejecting anything outside this set early is cheaper
# and clearer than letting the join silently return nothing.
VALID_METRIC_NAMES = {
    "cpu_percent", "mem_percent", "swap_percent",
    "disk_read_bytes", "disk_write_bytes", "disk_read_count", "disk_write_count",
    "net_bytes_sent", "net_bytes_recv", "net_packets_sent", "net_packets_recv",
    "load_avg_1", "load_avg_5", "load_avg_15", "thread_count_total",
}

# A predicted value is only matched against an actual sample within this
# many seconds of its target timestamp -- otherwise a forecast targeting
# a moment with no nearby actual sample (e.g. monitoring was stopped)
# would silently pair with a stale, unrelated reading. 5x a typical 1-2s
# polling interval comfortably covers normal jitter/gaps
# without matching across a real outage.
MAX_MATCH_DISTANCE_SECONDS = 10.0


@dataclass(frozen=True)
class PredictedVsActualPoint:
    metric_name: str
    horizon_label: str
    generated_at: float
    target_timestamp: float
    predicted_value: float
    actual_value: float | None
    error: float | None
    current_value: float | None
    confidence: float | None
    model_version: str


def _nearest(
    target_timestamp: float, sorted_timestamps: list[float], samples: list[SystemMetricSample]
) -> SystemMetricSample | None:
    """Nearest-timestamp lookup via binary search over pre-sorted actuals."""
    if not sorted_timestamps:
        return None
    idx = bisect.bisect_left(sorted_timestamps, target_timestamp)
    candidates = [i for i in (idx - 1, idx) if 0 <= i < len(sorted_timestamps)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(sorted_timestamps[i] - target_timestamp))
    if abs(sorted_timestamps[best] - target_timestamp) > MAX_MATCH_DISTANCE_SECONDS:
        return None
    return samples[best]


def join_predicted_vs_actual(
    engine: Engine,
    metric_name: str,
    start: float | None = None,
    end: float | None = None,
    horizon_label: str | None = None,
    limit: int | None = None,
) -> list[PredictedVsActualPoint]:
    """Join forecast predictions against the actual metric value nearest each ``target_timestamp``
    (and, separately, nearest each ``generated_at``, as ``current_value``).

    ``actual_value``/``error``/``current_value`` are ``None`` when no
    observed sample falls within ``MAX_MATCH_DISTANCE_SECONDS`` of the
    relevant moment (e.g. the target is in the near future and hasn't
    happened yet, or monitoring had a gap) -- a caller should treat that
    as "not yet available," not an error.
    """
    if metric_name not in VALID_METRIC_NAMES:
        raise ValueError(f"Unknown metric_name {metric_name!r}; expected one of {sorted(VALID_METRIC_NAMES)}")

    predictions = get_forecast_predictions_range(engine, start=start, end=end, metric_name=metric_name, limit=limit)
    if horizon_label is not None:
        predictions = [p for p in predictions if p.horizon_label == horizon_label]
    if not predictions:
        return []

    # Actuals need a window wide enough to cover both target_timestamps
    # (in the future relative to generated_at) and generated_ats (the
    # earlier reference point directional accuracy needs).
    all_timestamps = [p.target_timestamp for p in predictions] + [p.generated_at for p in predictions]
    actual_start = min(all_timestamps) - MAX_MATCH_DISTANCE_SECONDS
    actual_end = max(all_timestamps) + MAX_MATCH_DISTANCE_SECONDS
    actual_samples = get_system_metrics_range(engine, start=actual_start, end=actual_end)
    actual_samples.sort(key=lambda s: s.timestamp)
    sorted_timestamps = [s.timestamp for s in actual_samples]

    points: list[PredictedVsActualPoint] = []
    for pred in predictions:
        actual_sample = _nearest(pred.target_timestamp, sorted_timestamps, actual_samples)
        actual_value = getattr(actual_sample, metric_name) if actual_sample is not None else None
        error = None if actual_value is None else pred.predicted_value - actual_value

        current_sample = _nearest(pred.generated_at, sorted_timestamps, actual_samples)
        current_value = getattr(current_sample, metric_name) if current_sample is not None else None

        points.append(
            PredictedVsActualPoint(
                metric_name=metric_name,
                horizon_label=pred.horizon_label,
                generated_at=pred.generated_at,
                target_timestamp=pred.target_timestamp,
                predicted_value=pred.predicted_value,
                actual_value=actual_value,
                error=error,
                current_value=current_value,
                confidence=pred.confidence,
                model_version=pred.model_version,
            )
        )
    points.sort(key=lambda p: p.target_timestamp)
    return points
