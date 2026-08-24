from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine

from aiperfopt.config.settings import ForecastingSettings
from aiperfopt.features import FeatureEngineeringConfig
from aiperfopt.forecasting.persistence import artifact_exists
from aiperfopt.forecasting.training_pipeline import (
    generate_holdout_scenario,
    generate_training_scenarios,
    train_and_evaluate_all,
)
from aiperfopt.storage import get_evaluations_range


def _fast_forecasting_settings(models_dir: Path) -> ForecastingSettings:
    return ForecastingSettings(
        short_horizon_seconds=(10, 20), medium_horizon_seconds=(30, 60),
        walk_forward_splits=2, min_training_rows=15, models_dir=models_dir,
        directional_stability_threshold=0.02, baseline_improvement_margin=0.05,
    )


def _fast_feature_config() -> FeatureEngineeringConfig:
    return FeatureEngineeringConfig(lookback_windows_seconds=(10, 30), tolerance_seconds=3.0)


def test_full_training_pipeline_one_metric_end_to_end(storage_engine: Engine, tmp_path: Path):
    fconfig = _fast_feature_config()
    fsettings = _fast_forecasting_settings(tmp_path / "models")

    summary = train_and_evaluate_all(
        storage_engine, fconfig, fsettings, model_version="test-v1",
        target_metrics=["cpu_percent"], scenario_duration_seconds=150, interval_seconds=2.0,
    )

    assert len(summary.outcomes) == 2  # short + medium horizons
    for outcome in summary.outcomes:
        assert outcome.metric_name == "cpu_percent"
        assert outcome.selected_forecaster in (
            "lightgbm", "naive_persistence", "moving_average", "exponential_smoothing",
        )
        assert outcome.walk_forward_summary  # non-empty: every model was evaluated
        assert "lightgbm" in outcome.walk_forward_summary


def test_walk_forward_evaluation_rows_persisted_to_storage(storage_engine: Engine, tmp_path: Path):
    fconfig = _fast_feature_config()
    fsettings = _fast_forecasting_settings(tmp_path / "models")

    train_and_evaluate_all(
        storage_engine, fconfig, fsettings, model_version="test-v1",
        target_metrics=["cpu_percent"], scenario_duration_seconds=150, interval_seconds=2.0,
    )

    rows = get_evaluations_range(storage_engine)
    assert len(rows) > 0
    walk_forward_rows = [r for r in rows if r.dataset_tag == "walk_forward"]
    holdout_rows = [r for r in rows if r.dataset_tag.startswith("holdout_")]
    assert len(walk_forward_rows) > 0
    assert len(holdout_rows) > 0  # held-out generalization check (Section 11) ran and was recorded

    assert any(r.baseline_name is None and r.model_name == "lightgbm" for r in walk_forward_rows)
    assert any(r.baseline_name is not None for r in walk_forward_rows)


def test_lightgbm_artifact_saved_only_when_selected(storage_engine: Engine, tmp_path: Path):
    fconfig = _fast_feature_config()
    fsettings = _fast_forecasting_settings(tmp_path / "models")

    summary = train_and_evaluate_all(
        storage_engine, fconfig, fsettings, model_version="test-v1",
        target_metrics=["cpu_percent"], scenario_duration_seconds=150, interval_seconds=2.0,
    )

    for outcome in summary.outcomes:
        exists = artifact_exists(tmp_path / "models", "cpu_percent", outcome.horizon_label, "test-v1")
        if outcome.selected_forecaster == "lightgbm":
            assert exists is True
            assert outcome.artifact_path is not None
        else:
            assert exists is False
            assert outcome.artifact_path is None


def test_held_out_scenario_never_appears_in_walk_forward_training():
    """The held-out scenario generator uses a different time block than
    every training scenario — verifying this holds is what makes Section
    11's "held out entirely from all training folds" claim checkable."""
    scenarios = generate_training_scenarios(duration_seconds=100, interval_seconds=2.0)
    holdout = generate_holdout_scenario(
        duration_seconds=100, interval_seconds=2.0, scenario_count=len(scenarios)
    )

    training_timestamps = {s.timestamp for samples in scenarios.values() for s in samples}
    holdout_timestamps = {s.timestamp for s in holdout}
    assert training_timestamps.isdisjoint(holdout_timestamps)
