"""
Anomaly Detection calibration against real usage.

The Isolation Forest scores "how unlike training-time normal is this
row", which is only meaningful if training-time normal came from the
machine it is later asked to score. This module is the calibration path
for a detector meant to score live traffic: it reads this machine's own
collected feature vectors and refuses to fit if there isn't yet enough of
it (see :func:`build_calibration_matrix`).
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine

from aiperfopt.anomaly.detector import AnomalyDetector
from aiperfopt.common.exceptions import AnomalyDetectionError
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.features.engine import compute_feature_matrix
from aiperfopt.storage.reader import get_system_metrics_range

# Deliberately the same floor as AnomalyDetector's own default
# min_training_rows: below this, "real normal" is too thin a sample to
# calibrate against at all, real data or not.
DEFAULT_MIN_REAL_ROWS = 20


def build_calibration_matrix(
    engine: Engine,
    feature_config: FeatureEngineeringConfig,
    start: float | None = None,
    end: float | None = None,
    min_real_rows: int = DEFAULT_MIN_REAL_ROWS,
) -> pd.DataFrame:
    """Build the feature matrix a live-serving :class:`AnomalyDetector` must fit on.

    Reads this machine's own collected system metrics (``start``/``end``
    default to all available history), computes the same engineered
    feature matrix Forecasting and live inference use
    (:func:`aiperfopt.features.engine.compute_feature_matrix`), and
    raises :class:`AnomalyDetectionError` if fewer than ``min_real_rows``
    feature rows resulted. This is recoverable: it happens naturally
    before enough passive collection has accumulated.
    """
    real_samples = get_system_metrics_range(engine, start=start, end=end)
    real_matrix = compute_feature_matrix(real_samples, feature_config)

    if len(real_matrix) < min_real_rows:
        raise AnomalyDetectionError(
            f"Only {len(real_matrix)} real collected feature row(s) available; "
            f"need at least {min_real_rows} before an anomaly detector can be "
            "calibrated for live use. Keep the collector running and retry later.",
            recoverable=True,
            context={"real_rows": len(real_matrix), "min_real_rows": min_real_rows},
        )

    return real_matrix


def calibrate_anomaly_detector(
    engine: Engine,
    feature_config: FeatureEngineeringConfig,
    start: float | None = None,
    end: float | None = None,
    min_real_rows: int = DEFAULT_MIN_REAL_ROWS,
    detector: AnomalyDetector | None = None,
) -> AnomalyDetector:
    """Fit and return an :class:`AnomalyDetector` calibrated against real usage.

    Thin wrapper around :func:`build_calibration_matrix` +
    :meth:`AnomalyDetector.fit`. Callers preparing a detector for live
    scoring (as opposed to the offline training runs, which fit their own
    via :mod:`aiperfopt.contention.training_pipeline`) should use this
    rather than calling ``fit`` directly, so the real-data floor is
    always enforced.
    """
    matrix = build_calibration_matrix(
        engine, feature_config, start=start, end=end, min_real_rows=min_real_rows,
    )
    feature_cols = [c for c in matrix.columns if c != "timestamp"]

    detector = detector if detector is not None else AnomalyDetector()
    detector.fit(matrix[feature_cols])
    return detector
