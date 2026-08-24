"""
Anomaly detector artifact persistence.

Unlike LightGBM (:mod:`aiperfopt.forecasting.persistence`, which uses
LightGBM's own portable text format), scikit-learn's ``IsolationForest``
has no native serialization format. ``joblib`` is used instead — it's
the scikit-learn ecosystem's own documented, standard approach for
persisting fitted estimators, not a generic pickle-and-hope choice. The
same caveats as pickle still apply (version skew across scikit-learn
releases, no protection against loading untrusted files) — acceptable
here since artifacts are only ever produced and consumed by this same
codebase's own training pipeline, never loaded from an external/untrusted
source.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from aiperfopt.anomaly.detector import AnomalyDetector
from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import AnomalyDetectionError

logger = get_logger(__name__)

_ARTIFACT_FILENAME = "anomaly_detector.joblib"


def save_anomaly_detector(detector: AnomalyDetector, models_dir: Path, model_version: str) -> Path:
    if detector.model is None:
        raise AnomalyDetectionError("Cannot save an untrained anomaly detector", recoverable=False)

    target_dir = models_dir / "anomaly" / model_version
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / _ARTIFACT_FILENAME
    joblib.dump(detector, path)

    logger.info("Anomaly detector artifact saved", extra={"path": str(path)})
    return path


def load_anomaly_detector(models_dir: Path, model_version: str) -> AnomalyDetector:
    path = models_dir / "anomaly" / model_version / _ARTIFACT_FILENAME
    if not path.exists():
        raise AnomalyDetectionError(
            f"No saved anomaly detector artifact at {path}", recoverable=False, context={"path": str(path)}
        )
    return joblib.load(path)


def anomaly_artifact_exists(models_dir: Path, model_version: str) -> bool:
    return (models_dir / "anomaly" / model_version / _ARTIFACT_FILENAME).exists()
