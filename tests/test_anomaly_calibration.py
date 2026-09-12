from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import Engine

from aiperfopt.anomaly.calibration import (
    DEFAULT_MIN_REAL_ROWS,
    build_calibration_matrix,
    calibrate_anomaly_detector,
)
from aiperfopt.common.exceptions import AnomalyDetectionError
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.features.engine import compute_feature_matrix
from aiperfopt.storage.records import SystemMetricSample
from aiperfopt.storage.writer import insert_system_metric_samples_batch


def _idle_like_samples(n: int, interval: float = 3.0, start: float = 0.0, seed: int = 0) -> list[SystemMetricSample]:
    """A stationary, quiet-machine-like series -- real collected data's shape, not a stress scenario."""
    rng = np.random.default_rng(seed)
    return [
        SystemMetricSample(
            timestamp=start + i * interval,
            cpu_percent=float(np.clip(rng.normal(8.0, 1.5), 0, 100)),
            mem_percent=float(np.clip(rng.normal(45.0, 1.0), 0, 100)),
            swap_percent=0.0,
            load_avg_1=float(max(0.0, rng.normal(0.3, 0.05))),
            disk_read_bytes=int(1_000_000 + i * 1000),
            disk_write_bytes=int(500_000 + i * 500),
            net_bytes_sent=int(200_000 + i * 200),
            net_bytes_recv=int(300_000 + i * 300),
        )
        for i in range(n)
    ]


def test_calibration_raises_with_insufficient_real_rows(storage_engine: Engine):
    insert_system_metric_samples_batch(storage_engine, _idle_like_samples(n=5))
    config = FeatureEngineeringConfig()

    with pytest.raises(AnomalyDetectionError):
        calibrate_anomaly_detector(storage_engine, config, min_real_rows=DEFAULT_MIN_REAL_ROWS)


def test_calibration_raises_when_no_real_data_at_all(storage_engine: Engine):
    config = FeatureEngineeringConfig()

    with pytest.raises(AnomalyDetectionError):
        build_calibration_matrix(storage_engine, config, min_real_rows=DEFAULT_MIN_REAL_ROWS)


def test_calibration_fits_on_real_data_and_generalizes_to_holdout(storage_engine: Engine):
    """The two 'real normal data' success criteria from the Phase 3 plan.

    Trains on one stretch of real-shaped collected data, scores that same
    training matrix (should be typical, mean < 0.3), then scores a
    disjoint later stretch drawn from the identical stationary process
    (fewer than 10% should exceed the significance threshold) -- the
    generalization check the plan requires before a calibrated detector
    is trusted for live use.
    """
    config = FeatureEngineeringConfig()
    interval = 3.0

    train_samples = _idle_like_samples(n=700, interval=interval, start=0.0, seed=1)
    train_cutoff = train_samples[-1].timestamp
    holdout_samples = _idle_like_samples(
        n=150, interval=interval, start=train_cutoff + interval, seed=2
    )

    insert_system_metric_samples_batch(storage_engine, train_samples)

    detector = calibrate_anomaly_detector(storage_engine, config, end=train_cutoff, min_real_rows=20)

    train_matrix = build_calibration_matrix(storage_engine, config, end=train_cutoff, min_real_rows=20)
    feature_cols = [c for c in train_matrix.columns if c != "timestamp"]

    train_scores = detector.score_batch(train_matrix[feature_cols])
    assert train_scores.mean() < 0.3

    holdout_matrix = compute_feature_matrix(holdout_samples, config)
    holdout_scores = detector.score_batch(holdout_matrix[feature_cols])

    threshold = 0.7
    exceed_fraction = (holdout_scores >= threshold).mean()
    assert exceed_fraction < 0.10


def test_calibration_blends_synthetic_only_after_real_floor_met(storage_engine: Engine):
    """Synthetic rows may add capacity but can't substitute for the real floor."""
    config = FeatureEngineeringConfig()
    real_samples = _idle_like_samples(n=700, interval=3.0, start=0.0, seed=3)
    insert_system_metric_samples_batch(storage_engine, real_samples)

    real_matrix = build_calibration_matrix(storage_engine, config, min_real_rows=20)
    synthetic_matrix = real_matrix.iloc[:10].copy()

    combined = build_calibration_matrix(
        storage_engine, config, min_real_rows=20, synthetic_features=synthetic_matrix
    )
    assert len(combined) == len(real_matrix) + len(synthetic_matrix)
