"""Anomaly Detection (Section 7.1, item 7): Isolation Forest on engineered
feature vectors (Section 10.4, FR-7.1)."""

from aiperfopt.anomaly.detector import DEFAULT_ISOLATION_FOREST_PARAMS, AnomalyDetector
from aiperfopt.anomaly.notifications import emit_anomaly_notification_if_significant
from aiperfopt.anomaly.persistence import (
    anomaly_artifact_exists,
    load_anomaly_detector,
    save_anomaly_detector,
)

__all__ = [
    "DEFAULT_ISOLATION_FOREST_PARAMS",
    "AnomalyDetector",
    "anomaly_artifact_exists",
    "emit_anomaly_notification_if_significant",
    "load_anomaly_detector",
    "save_anomaly_detector",
]
