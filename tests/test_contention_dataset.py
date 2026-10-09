from __future__ import annotations

import pytest

from aiperfopt.anomaly.detector import AnomalyDetector
from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.contention.dataset import build_contention_dataset
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.features.engine import compute_feature_matrix
from aiperfopt.forecasting.baselines import NaivePersistenceForecaster
from aiperfopt.storage.records import SystemMetricSample


def _samples(n=100, interval=2.0, start=0.0, cpu_fn=None):
    cpu_fn = cpu_fn or (lambda i: 20.0)
    return [
        SystemMetricSample(timestamp=start + i * interval, cpu_percent=cpu_fn(i), mem_percent=30.0)
        for i in range(n)
    ]


def _fitted_anomaly_detector(samples, feature_config):
    fm = compute_feature_matrix(samples, feature_config)
    detector = AnomalyDetector(min_training_rows=20)
    detector.fit(fm[[c for c in fm.columns if c != "timestamp"]])
    return detector


@pytest.mark.behavioural
def test_dataset_has_expected_columns():
    feature_config = FeatureEngineeringConfig(lookback_windows_seconds=(10, 30), tolerance_seconds=3.0)
    samples = _samples(100)
    detector = _fitted_anomaly_detector(samples, feature_config)
    labeling = ContentionLabelingConfig(value_thresholds={"cpu_percent": 80.0}, tolerance_seconds=3.0)

    ds = build_contention_dataset(
        samples, {"cpu_percent": NaivePersistenceForecaster()}, detector, feature_config, labeling,
        horizon_seconds=10,
    )

    assert "forecast_cpu_percent" in ds.columns
    assert "current_cpu_percent" in ds.columns
    assert "anomaly_score" in ds.columns
    assert "label" in ds.columns
    # The raw engineered feature matrix's own columns must NOT leak
    # through directly — the "not raw metrics directly" boundary.
    assert "cpu_percent_roll_mean_10s" not in ds.columns


@pytest.mark.behavioural
def test_label_is_one_when_metric_crosses_threshold():
    feature_config = FeatureEngineeringConfig(lookback_windows_seconds=(10, 30), tolerance_seconds=3.0)
    # cpu_percent ramps well past the threshold partway through.
    samples = _samples(100, cpu_fn=lambda i: 10.0 if i < 50 else 95.0)
    detector = _fitted_anomaly_detector(samples, feature_config)
    labeling = ContentionLabelingConfig(value_thresholds={"cpu_percent": 80.0}, tolerance_seconds=3.0)

    ds = build_contention_dataset(
        samples, {"cpu_percent": NaivePersistenceForecaster()}, detector, feature_config, labeling,
        horizon_seconds=10,
    )

    assert ds["label"].sum() > 0
    assert set(ds["label"].unique()).issubset({0, 1})


@pytest.mark.behavioural
def test_label_is_always_zero_when_never_crossing_threshold():
    feature_config = FeatureEngineeringConfig(lookback_windows_seconds=(10, 30), tolerance_seconds=3.0)
    samples = _samples(100, cpu_fn=lambda i: 15.0)  # always well below threshold
    detector = _fitted_anomaly_detector(samples, feature_config)
    labeling = ContentionLabelingConfig(value_thresholds={"cpu_percent": 80.0}, tolerance_seconds=3.0)

    ds = build_contention_dataset(
        samples, {"cpu_percent": NaivePersistenceForecaster()}, detector, feature_config, labeling,
        horizon_seconds=10,
    )

    assert ds["label"].sum() == 0


@pytest.mark.behavioural
def test_multiple_target_metrics_produce_columns_for_each():
    feature_config = FeatureEngineeringConfig(lookback_windows_seconds=(10, 30), tolerance_seconds=3.0)
    samples = _samples(100)
    detector = _fitted_anomaly_detector(samples, feature_config)
    labeling = ContentionLabelingConfig(
        value_thresholds={"cpu_percent": 80.0, "mem_percent": 90.0}, tolerance_seconds=3.0
    )

    ds = build_contention_dataset(
        samples,
        {"cpu_percent": NaivePersistenceForecaster(), "mem_percent": NaivePersistenceForecaster()},
        detector, feature_config, labeling, horizon_seconds=10,
    )

    assert "forecast_cpu_percent" in ds.columns
    assert "forecast_mem_percent" in ds.columns


@pytest.mark.plumbing
def test_empty_samples_returns_empty_dataset():
    feature_config = FeatureEngineeringConfig()
    labeling = ContentionLabelingConfig()
    detector = AnomalyDetector()  # untrained, but score_batch never called for empty input
    ds = build_contention_dataset(
        [], {"cpu_percent": NaivePersistenceForecaster()}, detector, feature_config, labeling,
        horizon_seconds=10,
    )
    assert ds.empty


@pytest.mark.plumbing
def test_no_deployed_forecasters_returns_empty_dataset():
    feature_config = FeatureEngineeringConfig()
    labeling = ContentionLabelingConfig()
    samples = _samples(50)
    detector = _fitted_anomaly_detector(samples, feature_config)
    ds = build_contention_dataset(samples, {}, detector, feature_config, labeling, horizon_seconds=10)
    assert ds.empty
