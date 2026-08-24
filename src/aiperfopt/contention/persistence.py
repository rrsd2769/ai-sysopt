"""Contention classifier artifact persistence — same joblib rationale as
:mod:`aiperfopt.anomaly.persistence` (scikit-learn has no portable native
format the way LightGBM does)."""

from __future__ import annotations

from pathlib import Path

import joblib

from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import ContentionPredictionError
from aiperfopt.contention.model import ContentionClassifier

logger = get_logger(__name__)

_ARTIFACT_FILENAME = "contention_classifier.joblib"


def save_contention_classifier(
    classifier: ContentionClassifier, models_dir: Path, model_version: str
) -> Path:
    if classifier.model is None:
        raise ContentionPredictionError("Cannot save an untrained classifier", recoverable=False)

    target_dir = models_dir / "contention" / model_version
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / _ARTIFACT_FILENAME
    joblib.dump(classifier, path)

    logger.info("Contention classifier artifact saved", extra={"path": str(path)})
    return path


def load_contention_classifier(models_dir: Path, model_version: str) -> ContentionClassifier:
    path = models_dir / "contention" / model_version / _ARTIFACT_FILENAME
    if not path.exists():
        raise ContentionPredictionError(
            f"No saved contention classifier artifact at {path}", recoverable=False,
            context={"path": str(path)},
        )
    return joblib.load(path)


def contention_artifact_exists(models_dir: Path, model_version: str) -> bool:
    return (models_dir / "contention" / model_version / _ARTIFACT_FILENAME).exists()
