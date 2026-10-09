"""Anomaly Detection: Isolation Forest on engineered
feature vectors."""

from aiperfopt.anomaly.calibration import build_calibration_matrix, calibrate_anomaly_detector
from aiperfopt.anomaly.detector import ANOMALY_ARTIFACT_SCHEMA_VERSION, DEFAULT_ISOLATION_FOREST_PARAMS, AnomalyDetector
from aiperfopt.anomaly.notifications import emit_anomaly_notification_if_significant
from aiperfopt.anomaly.persistence import (
    anomaly_artifact_exists,
    load_anomaly_detector,
    save_anomaly_detector,
)

__all__ = [
    "ANOMALY_ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_ISOLATION_FOREST_PARAMS",
    "AnomalyDetector",
    "anomaly_artifact_exists",
    "build_calibration_matrix",
    "calibrate_anomaly_detector",
    "emit_anomaly_notification_if_significant",
    "load_anomaly_detector",
    "save_anomaly_detector",
]
