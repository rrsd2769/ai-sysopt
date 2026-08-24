from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine

from aiperfopt.config.settings import ContentionSettings, ForecastingSettings
from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.contention.training_pipeline import (
    _interleave_scenario_chunks,
    train_contention_and_anomaly_pipeline,
)
from aiperfopt.features import FeatureEngineeringConfig
from aiperfopt.forecasting.persistence import artifact_exists as forecasting_artifact_exists
from aiperfopt.forecasting.training_pipeline import train_and_evaluate_all
from aiperfopt.anomaly.persistence import anomaly_artifact_exists
from aiperfopt.contention.persistence import contention_artifact_exists
from aiperfopt.storage import get_evaluations_range


def _fast_settings(models_dir: Path):
    fconfig = FeatureEngineeringConfig(lookback_windows_seconds=(10, 30), tolerance_seconds=3.0)
    fsettings = ForecastingSettings(
        short_horizon_seconds=(10, 20), medium_horizon_seconds=(30, 60),
        walk_forward_splits=2, min_training_rows=15, models_dir=models_dir,
        directional_stability_threshold=0.02, baseline_improvement_margin=0.05,
    )
    csettings = ContentionSettings(
        decision_threshold=0.5, anomaly_dampening_factor=0.3,
        anomaly_significant_threshold=0.7, contributing_processes_top_n=3,
    )
    labeling_config = ContentionLabelingConfig(
        value_thresholds={"cpu_percent": 60.0}, tolerance_seconds=3.0
    )
    return fconfig, fsettings, csettings, labeling_config


def test_full_forecasting_then_contention_pipeline(storage_engine: Engine, tmp_path: Path):
    fconfig, fsettings, csettings, labeling_config = _fast_settings(tmp_path / "models")

    forecast_summary = train_and_evaluate_all(
        storage_engine, fconfig, fsettings, model_version="test-v1",
        target_metrics=["cpu_percent"], scenario_duration_seconds=150, interval_seconds=2.0,
    )
    short_outcomes = {o.metric_name: o for o in forecast_summary.outcomes if o.horizon_label == "short"}
    assert "cpu_percent" in short_outcomes

    summary = train_contention_and_anomaly_pipeline(
        storage_engine, fconfig, fsettings, csettings, labeling_config, short_outcomes,
        model_version="test-v1", scenario_duration_seconds=150, interval_seconds=2.0,
    )

    assert summary.n_training_rows > 0
    assert anomaly_artifact_exists(tmp_path / "models", "test-v1")


def test_contention_evaluations_persisted_with_extra_metrics(storage_engine: Engine, tmp_path: Path):
    fconfig, fsettings, csettings, labeling_config = _fast_settings(tmp_path / "models")

    forecast_summary = train_and_evaluate_all(
        storage_engine, fconfig, fsettings, model_version="test-v1",
        target_metrics=["cpu_percent"], scenario_duration_seconds=150, interval_seconds=2.0,
    )
    short_outcomes = {o.metric_name: o for o in forecast_summary.outcomes if o.horizon_label == "short"}

    train_contention_and_anomaly_pipeline(
        storage_engine, fconfig, fsettings, csettings, labeling_config, short_outcomes,
        model_version="test-v1", scenario_duration_seconds=150, interval_seconds=2.0,
    )

    rows = get_evaluations_range(storage_engine)
    contention_rows = [r for r in rows if r.model_name == "contention_classifier"]
    assert len(contention_rows) > 0
    # Classification metrics live in `extra`; mae/rmse stay null (not
    # applicable to a classifier) — verifying the schema-reuse design.
    for row in contention_rows:
        assert row.mae is None
        assert row.rmse is None
        assert row.extra is not None
        assert "precision" in row.extra or "selected_forecaster" in row.extra


def test_classifier_artifact_saved_when_training_succeeds(storage_engine: Engine, tmp_path: Path):
    fconfig, fsettings, csettings, labeling_config = _fast_settings(tmp_path / "models")

    forecast_summary = train_and_evaluate_all(
        storage_engine, fconfig, fsettings, model_version="test-v1",
        target_metrics=["cpu_percent"], scenario_duration_seconds=150, interval_seconds=2.0,
    )
    short_outcomes = {o.metric_name: o for o in forecast_summary.outcomes if o.horizon_label == "short"}

    summary = train_contention_and_anomaly_pipeline(
        storage_engine, fconfig, fsettings, csettings, labeling_config, short_outcomes,
        model_version="test-v1", scenario_duration_seconds=150, interval_seconds=2.0,
    )

    if summary.classifier_artifact_path is not None:
        assert contention_artifact_exists(tmp_path / "models", "test-v1")


def test_no_deployed_forecasters_raises_value_error(storage_engine: Engine, tmp_path: Path):
    fconfig, fsettings, csettings, labeling_config = _fast_settings(tmp_path / "models")
    import pytest

    with pytest.raises(ValueError):
        train_contention_and_anomaly_pipeline(
            storage_engine, fconfig, fsettings, csettings, labeling_config, {},
            model_version="test-v1", scenario_duration_seconds=150, interval_seconds=2.0,
        )


def test_interleave_scenario_chunks_preserves_all_rows():
    import pandas as pd

    ds_a = pd.DataFrame({"timestamp": range(10), "label": [0] * 10})
    ds_b = pd.DataFrame({"timestamp": range(100, 108), "label": [1] * 8})

    combined = _interleave_scenario_chunks([ds_a, ds_b], n_chunks=4)

    assert len(combined) == len(ds_a) + len(ds_b)
    assert set(combined["timestamp"]) == set(ds_a["timestamp"]) | set(ds_b["timestamp"])


def test_interleave_scenario_chunks_mixes_scenarios_in_early_rows():
    import pandas as pd

    ds_a = pd.DataFrame({"timestamp": range(0, 100), "source": ["a"] * 100})
    ds_b = pd.DataFrame({"timestamp": range(1000, 1100), "source": ["b"] * 100})

    combined = _interleave_scenario_chunks([ds_a, ds_b], n_chunks=4)

    # With block concatenation, the first 25 rows would all be source "a".
    # With interleaving, both sources should appear well before row 50.
    first_quarter_sources = set(combined["source"].iloc[:60])
    assert first_quarter_sources == {"a", "b"}


def test_interleave_scenario_chunks_empty_input():
    result = _interleave_scenario_chunks([], n_chunks=4)
    assert result.empty
