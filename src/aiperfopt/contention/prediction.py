"""
Live contention prediction: the classifier's raw probability, adjusted by
Anomaly Detection's score, resolved into contributing metrics/processes,
and packaged as a storable :class:`ContentionPredictionRecord`.

Anomaly-score dampening (FR-7.2)
------------------------------------
FR-7.2 requires anomaly output to "modulate contention-prediction
confidence." The ``contention_predictions`` schema (Phase 1) has no
separate confidence column distinct from ``probability`` — so probability
itself is the carrier: ``adjusted = raw * (1 - dampening_factor *
anomaly_score)``. A highly anomalous reading (the classifier is
extrapolating well outside its training distribution) pulls the reported
probability toward zero rather than trusting it at face value; a
completely typical reading (anomaly_score == 0) leaves it untouched.
``anomaly_score`` is still stored alongside the adjusted probability, so
the adjustment is always auditable, not silently baked in.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine

from aiperfopt.contention.model import ContentionClassifier
from aiperfopt.contention.processes import get_contributing_processes
from aiperfopt.storage.records import ContentionPredictionRecord


def _strip_feature_prefix(feature_name: str) -> str:
    for prefix in ("forecast_", "current_"):
        if feature_name.startswith(prefix):
            return feature_name[len(prefix):]
    return feature_name


def predict_contention(
    classifier: ContentionClassifier,
    engine: Engine,
    feature_row: pd.Series,
    as_of: float,
    decision_threshold: float,
    anomaly_dampening_factor: float,
    contributing_processes_top_n: int,
    model_version: str,
    contributing_metrics_top_n: int = 5,
) -> ContentionPredictionRecord:
    """Produce one contention prediction ready for storage.

    ``feature_row`` must contain the classifier's expected
    ``forecast_{metric}``/``current_{metric}`` columns plus
    ``anomaly_score`` — i.e. a row shaped like
    :func:`aiperfopt.contention.dataset.build_contention_dataset`'s output
    (minus ``label``, which live inference obviously doesn't have).
    """
    raw_probability = classifier.predict_proba(feature_row)
    anomaly_score = float(feature_row.get("anomaly_score", 0.0))

    adjusted_probability = raw_probability * (1.0 - anomaly_dampening_factor * anomaly_score)
    adjusted_probability = max(0.0, min(1.0, adjusted_probability))
    triggered = adjusted_probability >= decision_threshold

    contributing = classifier.contributing_metrics(feature_row, top_n=contributing_metrics_top_n)
    contributing_metrics_payload = [
        {"metric": c.metric_name, "contribution": c.contribution} for c in contributing
    ]

    contributing_processes_payload: list[dict] = []
    # Only resolve contributing processes for metric-level features (not
    # anomaly_score itself, which isn't a per-process-attributable metric).
    metric_level_contributors = [c for c in contributing if c.metric_name != "anomaly_score"]
    if metric_level_contributors:
        dominant_metric = _strip_feature_prefix(metric_level_contributors[0].metric_name)
        processes = get_contributing_processes(
            engine, as_of, dominant_metric, top_n=contributing_processes_top_n
        )
        contributing_processes_payload = [
            {"pid": p.pid, "name": p.name, "value": p.value} for p in processes
        ]

    return ContentionPredictionRecord(
        generated_at=as_of,
        probability=adjusted_probability,
        threshold_used=decision_threshold,
        triggered=triggered,
        contributing_metrics=contributing_metrics_payload,
        contributing_processes=contributing_processes_payload,
        anomaly_score=anomaly_score,
        model_version=model_version,
    )
