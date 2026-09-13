from __future__ import annotations

import math
from pathlib import Path

import pytest
from sqlalchemy import Engine

from aiperfopt.config.settings import ContentionSettings, ForecastingSettings
from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.contention.dataset import build_live_contention_input_row
from aiperfopt.contention.persistence import load_contention_classifier
from aiperfopt.contention.prediction import predict_contention
from aiperfopt.contention.training_pipeline import train_contention_and_anomaly_pipeline
from aiperfopt.anomaly.persistence import load_anomaly_detector
from aiperfopt.features import FeatureEngineeringConfig
from aiperfopt.features.storage_bridge import compute_live_feature_vector
from aiperfopt.forecasting.deployment import resolve_deployed_forecasters
from aiperfopt.forecasting.training_pipeline import train_and_evaluate_all
from aiperfopt.storage import insert_system_metric_samples_batch
from aiperfopt.storage.reader import get_recent_system_metrics
from aiperfopt.storage.records import SystemMetricSample


def _fast_settings(tmp_path: Path):
    fconfig = FeatureEngineeringConfig(lookback_windows_seconds=(10, 30), tolerance_seconds=3.0)
    fsettings = ForecastingSettings(
        short_horizon_seconds=(10, 20), medium_horizon_seconds=(30, 60),
        walk_forward_splits=2, min_training_rows=15, models_dir=tmp_path / "models",
        directional_stability_threshold=0.02, baseline_improvement_margin=0.05,
    )
    csettings = ContentionSettings(
        decision_threshold=0.5, anomaly_dampening_factor=0.3,
        anomaly_significant_threshold=0.7, contributing_processes_top_n=3,
    )
    labeling_config = ContentionLabelingConfig(value_thresholds={"cpu_percent": 60.0}, tolerance_seconds=3.0)
    return fconfig, fsettings, csettings, labeling_config


def _rising_cpu_samples(n: int, interval: float = 2.0) -> list[SystemMetricSample]:
    return [
        SystemMetricSample(
            timestamp=i * interval,
            cpu_percent=max(0.0, min(95.0, 5.0 + (i / n) * 90.0 + 4.0 * math.sin(i / 3.0))),
            mem_percent=50.0,
        )
        for i in range(n)
    ]


@pytest.mark.plumbing
def test_resolve_deployed_forecasters_reconstructs_from_persisted_evaluations(
    storage_engine: Engine, tmp_path: Path
):
    fconfig, fsettings, csettings, labeling_config = _fast_settings(tmp_path)
    insert_system_metric_samples_batch(storage_engine, _rising_cpu_samples(300))

    train_and_evaluate_all(
        storage_engine, fconfig, fsettings, model_version="v1", target_metrics=["cpu_percent"]
    )

    forecasters = resolve_deployed_forecasters(
        storage_engine, fconfig, fsettings, "v1", horizon_label="short", target_metrics=["cpu_percent"]
    )

    assert "cpu_percent" in forecasters
    assert forecasters["cpu_percent"].model_version is not None


@pytest.mark.behavioural
def test_resolve_deployed_forecasters_skips_metric_with_no_holdout_evaluation(
    storage_engine: Engine, tmp_path: Path
):
    fconfig, fsettings, csettings, labeling_config = _fast_settings(tmp_path)
    insert_system_metric_samples_batch(storage_engine, _rising_cpu_samples(300))

    train_and_evaluate_all(
        storage_engine, fconfig, fsettings, model_version="v1", target_metrics=["cpu_percent"]
    )

    # "mem_percent" was never trained under this model_version.
    forecasters = resolve_deployed_forecasters(
        storage_engine, fconfig, fsettings, "v1", horizon_label="short",
        target_metrics=["cpu_percent", "mem_percent"],
    )

    assert set(forecasters.keys()) == {"cpu_percent"}


@pytest.mark.plumbing
def test_resolve_deployed_forecasters_empty_for_unknown_version(storage_engine: Engine, tmp_path: Path):
    fconfig, fsettings, _, _ = _fast_settings(tmp_path)
    forecasters = resolve_deployed_forecasters(
        storage_engine, fconfig, fsettings, "no-such-version", target_metrics=["cpu_percent"]
    )
    assert forecasters == {}


@pytest.mark.plumbing
def test_build_live_contention_input_row_feeds_a_real_prediction(storage_engine: Engine, tmp_path: Path):
    """End-to-end: real trained forecaster + classifier + anomaly detector,
    fed a live row built from real stored history, produces a real
    contention prediction — the exact path Phase 9's live-inference loop
    exercises every cycle."""
    fconfig, fsettings, csettings, labeling_config = _fast_settings(tmp_path)
    insert_system_metric_samples_batch(storage_engine, _rising_cpu_samples(300))

    forecast_summary = train_and_evaluate_all(
        storage_engine, fconfig, fsettings, model_version="v1", target_metrics=["cpu_percent"]
    )
    short_outcomes = {
        o.metric_name: o for o in forecast_summary.outcomes
        if o.horizon_label == "short" and o.selected_forecaster != "none"
    }
    train_contention_and_anomaly_pipeline(
        storage_engine, fconfig, fsettings, csettings, labeling_config,
        short_outcomes, model_version="v1",
    )

    forecasters = resolve_deployed_forecasters(
        storage_engine, fconfig, fsettings, "v1", horizon_label="short", target_metrics=["cpu_percent"]
    )
    classifier = load_contention_classifier(fsettings.models_dir, "v1")
    detector = load_anomaly_detector(fsettings.models_dir, "v1")

    as_of = 598.0
    recent = get_recent_system_metrics(storage_engine, window_seconds=fconfig.min_history_seconds, as_of=as_of)
    feature_vector = compute_live_feature_vector(storage_engine, fconfig, as_of=as_of)
    assert feature_vector is not None

    row = build_live_contention_input_row(recent, feature_vector, forecasters, detector, fconfig)

    assert row is not None
    assert "forecast_cpu_percent" in row
    assert "current_cpu_percent" in row
    assert "confidence_cpu_percent" in row  # Phase 7: persisted alongside the forecast
    assert "anomaly_score" in row
    assert 0.0 <= row["anomaly_score"] <= 1.0

    prediction = predict_contention(
        classifier, storage_engine, row, as_of=feature_vector.timestamp, decision_threshold=0.5,
        anomaly_dampening_factor=0.3, contributing_processes_top_n=3, model_version="v1",
    )
    assert 0.0 <= prediction.probability <= 1.0
    assert isinstance(prediction.triggered, bool)


@pytest.mark.plumbing
def test_build_live_contention_input_row_returns_none_with_no_forecasters(storage_engine: Engine, tmp_path: Path):
    fconfig, fsettings, csettings, labeling_config = _fast_settings(tmp_path)
    insert_system_metric_samples_batch(storage_engine, _rising_cpu_samples(300))

    from aiperfopt.anomaly.detector import AnomalyDetector

    feature_vector = compute_live_feature_vector(storage_engine, fconfig, as_of=598.0)
    recent = get_recent_system_metrics(storage_engine, window_seconds=fconfig.min_history_seconds, as_of=598.0)

    row = build_live_contention_input_row(recent, feature_vector, {}, AnomalyDetector(), fconfig)
    assert row is None


@pytest.mark.behavioural
def test_build_forecast_records_coerces_missing_confidence_and_uses_settings_horizon(tmp_path: Path):
    """A deployed *baseline* forecaster (Section 10.7's contingency plan) has
    no ``confidence``/``horizon_seconds`` attribute of its own -- Phase 7's
    persistence must read those from the row / settings, not the forecaster
    object, and turn pandas' NaN-for-missing-confidence back into ``None``."""
    from types import SimpleNamespace

    import pandas as pd

    from aiperfopt.api.live_state import LiveInferenceLoop
    from aiperfopt.config.settings import ForecastingSettings

    fsettings = ForecastingSettings(
        short_horizon_seconds=(10, 20), medium_horizon_seconds=(30, 60),
        walk_forward_splits=2, min_training_rows=15, models_dir=tmp_path / "models",
        directional_stability_threshold=0.02, baseline_improvement_margin=0.05,
    )
    stub_loop = SimpleNamespace(forecasting_settings=fsettings)

    row = pd.Series({
        "forecast_cpu_percent": 55.0, "confidence_cpu_percent": math.nan,
        "forecast_mem_percent": 40.0, "confidence_mem_percent": 0.8,
    })

    records = LiveInferenceLoop._build_forecast_records(
        stub_loop, row, generated_at=1000.0, metric_names=["cpu_percent", "mem_percent"], model_version="v1"
    )

    by_metric = {r.metric_name: r for r in records}
    assert by_metric["cpu_percent"].confidence is None
    assert by_metric["mem_percent"].confidence == pytest.approx(0.8)
    for record in records:
        assert record.horizon_label == "short"
        assert record.horizon_seconds == fsettings.short_horizon_target_seconds
        assert record.target_timestamp == pytest.approx(1000.0 + fsettings.short_horizon_target_seconds)
        assert record.model_version == "v1"
