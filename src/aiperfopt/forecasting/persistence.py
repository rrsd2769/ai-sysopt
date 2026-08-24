"""
Model artifact persistence.

Trained models are not database rows — they're files on disk under
``AppSettings.forecasting.models_dir``, one subdirectory per
metric/horizon. This mirrors how Retraining (Phase 8, FR-11.3) is
expected to "preserve the previous model artifact on retrain" for
rollback: that's a filesystem operation on these same paths, not a
database concern.

LightGBM's own text format (``Booster.save_model`` /
``lgb.Booster(model_file=...)``) is used instead of pickling the whole
:class:`MetricForecaster` — it's portable across LightGBM versions and
avoids pickle's well-known fragility (version skew, arbitrary code
execution on load) for an artifact that may need to survive a retraining
cycle or two. A small JSON sidecar carries everything pickling would
otherwise be needed for: which columns the booster expects (in order),
the derived confidence scalar, and identifying metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb

from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import ForecastingError
from aiperfopt.forecasting.model import MetricForecaster

logger = get_logger(__name__)

_METADATA_FILENAME = "metadata.json"
_MODEL_FILENAME = "model.txt"


def _artifact_dir(models_dir: Path, metric_name: str, horizon_label: str, model_version: str) -> Path:
    return models_dir / metric_name / horizon_label / model_version


def save_forecaster(forecaster: MetricForecaster, models_dir: Path) -> Path:
    """Persist a trained forecaster. Returns the artifact directory."""
    if forecaster.booster is None:
        raise ForecastingError("Cannot save an untrained forecaster", recoverable=False)

    target_dir = _artifact_dir(
        models_dir, forecaster.metric_name, forecaster.horizon_label, forecaster.model_version
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    forecaster.booster.save_model(str(target_dir / _MODEL_FILENAME))
    metadata = {
        "metric_name": forecaster.metric_name,
        "horizon_label": forecaster.horizon_label,
        "horizon_seconds": forecaster.horizon_seconds,
        "model_version": forecaster.model_version,
        "feature_columns": forecaster.feature_columns,
        "confidence": forecaster.confidence,
    }
    (target_dir / _METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))

    logger.info(
        "Forecaster artifact saved",
        extra={"metric": forecaster.metric_name, "horizon": forecaster.horizon_label, "path": str(target_dir)},
    )
    return target_dir


def load_forecaster(
    models_dir: Path, metric_name: str, horizon_label: str, model_version: str
) -> MetricForecaster:
    """Load a previously-saved forecaster artifact."""
    target_dir = _artifact_dir(models_dir, metric_name, horizon_label, model_version)
    metadata_path = target_dir / _METADATA_FILENAME
    model_path = target_dir / _MODEL_FILENAME

    if not metadata_path.exists() or not model_path.exists():
        raise ForecastingError(
            f"No saved forecaster artifact at {target_dir}", recoverable=False,
            context={"path": str(target_dir)},
        )

    metadata = json.loads(metadata_path.read_text())
    forecaster = MetricForecaster(
        metric_name=metadata["metric_name"],
        horizon_label=metadata["horizon_label"],
        horizon_seconds=metadata["horizon_seconds"],
        model_version=metadata["model_version"],
    )
    forecaster.booster = lgb.Booster(model_file=str(model_path))
    forecaster.feature_columns = metadata["feature_columns"]
    forecaster.confidence = metadata["confidence"]
    return forecaster


def artifact_exists(models_dir: Path, metric_name: str, horizon_label: str, model_version: str) -> bool:
    target_dir = _artifact_dir(models_dir, metric_name, horizon_label, model_version)
    return (target_dir / _METADATA_FILENAME).exists() and (target_dir / _MODEL_FILENAME).exists()
