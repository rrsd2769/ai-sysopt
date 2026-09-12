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

from aiperfopt.anomaly.detector import ANOMALY_ARTIFACT_SCHEMA_VERSION, AnomalyDetector
from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import AnomalyDetectionError
from aiperfopt.features.config import FEATURE_SET_VERSION

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
    detector = joblib.load(path)

    # A pre-Phase-3 artifact (pickled before ``schema_version``/
    # ``_train_means`` existed on the class) unpickles with those
    # attributes simply absent from its __dict__ — it would silently
    # fall back to score()'s old 0.0-fill behavior (F1) if allowed to
    # load. Reject it outright rather than let it score incorrectly.
    artifact_version = getattr(detector, "schema_version", None)
    if artifact_version != ANOMALY_ARTIFACT_SCHEMA_VERSION:
        raise AnomalyDetectionError(
            f"Anomaly detector artifact at {path} has schema_version {artifact_version!r}, "
            f"expected {ANOMALY_ARTIFACT_SCHEMA_VERSION!r} — retrain to produce a compatible artifact.",
            recoverable=False,
            context={"path": str(path), "artifact_version": artifact_version},
        )

    # A pre-Phase-4 artifact (fit before the process-summary gauge
    # columns existed) unpickles with ``feature_set_version`` absent from
    # its __dict__, same class-attribute-default trap as above — see
    # ``_UNVERSIONED_FEATURE_SET`` in detector.py. Reject it rather than
    # let ``_prepare``'s reindex silently drop every v2-only column a
    # live v2 matrix now offers.
    artifact_feature_set = getattr(detector, "feature_set_version", None)
    if artifact_feature_set != FEATURE_SET_VERSION:
        raise AnomalyDetectionError(
            f"Anomaly detector artifact at {path} has feature_set_version {artifact_feature_set!r}, "
            f"expected {FEATURE_SET_VERSION!r} — retrain to produce a compatible artifact.",
            recoverable=False,
            context={"path": str(path), "artifact_feature_set_version": artifact_feature_set},
        )
    return detector


def anomaly_artifact_exists(models_dir: Path, model_version: str) -> bool:
    return (models_dir / "anomaly" / model_version / _ARTIFACT_FILENAME).exists()
