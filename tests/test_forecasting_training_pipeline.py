from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine

from aiperfopt.config.settings import ForecastingSettings
from aiperfopt.features import FeatureEngineeringConfig
from aiperfopt.forecasting.persistence import artifact_exists
from aiperfopt.forecasting.training_pipeline import train_and_evaluate_all
from aiperfopt.storage import get_evaluations_range
from tests.factories import make_training_data

HOLDOUT_TAG = "holdout_test"


def _data_kwargs() -> dict:
    scenarios, holdout = make_training_data()
    return {"scenarios": scenarios, "holdout_samples": holdout, "holdout_dataset_tag": HOLDOUT_TAG}


def _fast_forecasting_settings(models_dir: Path) -> ForecastingSettings:
    return ForecastingSettings(
        short_horizon_seconds=(10, 20), medium_horizon_seconds=(30, 60),
        walk_forward_splits=2, min_training_rows=15, models_dir=models_dir,
        directional_stability_threshold=0.02, baseline_improvement_margin=0.05,
    )


def _fast_feature_config() -> FeatureEngineeringConfig:
    return FeatureEngineeringConfig(lookback_windows_seconds=(10, 30), tolerance_seconds=3.0)


@pytest.mark.plumbing
def test_full_training_pipeline_one_metric_end_to_end(storage_engine: Engine, tmp_path: Path):
    fconfig = _fast_feature_config()
    fsettings = _fast_forecasting_settings(tmp_path / "models")

    summary = train_and_evaluate_all(
        storage_engine, fconfig, fsettings, model_version="test-v1",
        target_metrics=["cpu_percent"], **_data_kwargs(),
    )

    assert len(summary.outcomes) == 2  # short + medium horizons
    for outcome in summary.outcomes:
        assert outcome.metric_name == "cpu_percent"
        assert outcome.selected_forecaster in (
            "lightgbm", "naive_persistence", "moving_average", "exponential_smoothing",
        )
        assert outcome.walk_forward_summary  # non-empty: every model was evaluated
        assert "lightgbm" in outcome.walk_forward_summary


@pytest.mark.plumbing
def test_walk_forward_evaluation_rows_persisted_to_storage(storage_engine: Engine, tmp_path: Path):
    fconfig = _fast_feature_config()
    fsettings = _fast_forecasting_settings(tmp_path / "models")

    train_and_evaluate_all(
        storage_engine, fconfig, fsettings, model_version="test-v1",
        target_metrics=["cpu_percent"], **_data_kwargs(),
    )

    rows = get_evaluations_range(storage_engine)
    assert len(rows) > 0
    walk_forward_rows = [r for r in rows if r.dataset_tag == "walk_forward"]
    holdout_rows = [r for r in rows if r.dataset_tag.startswith("holdout_")]
    assert len(walk_forward_rows) > 0
    assert len(holdout_rows) > 0  # held-out generalization check (Section 11) ran and was recorded

    assert any(r.baseline_name is None and r.model_name == "lightgbm" for r in walk_forward_rows)
    assert any(r.baseline_name is not None for r in walk_forward_rows)


@pytest.mark.behavioural
def test_lightgbm_artifact_saved_only_when_selected(storage_engine: Engine, tmp_path: Path):
    fconfig = _fast_feature_config()
    fsettings = _fast_forecasting_settings(tmp_path / "models")

    summary = train_and_evaluate_all(
        storage_engine, fconfig, fsettings, model_version="test-v1",
        target_metrics=["cpu_percent"], **_data_kwargs(),
    )

    for outcome in summary.outcomes:
        exists = artifact_exists(tmp_path / "models", "cpu_percent", outcome.horizon_label, "test-v1")
        if outcome.selected_forecaster == "lightgbm":
            assert exists is True
            assert outcome.artifact_path is not None
        else:
            assert exists is False
            assert outcome.artifact_path is None


@pytest.mark.behavioural
def test_holdout_is_strictly_later_than_training():
    """The holdout slice is held out entirely from the training folds, which
    is what makes the generalization check meaningful."""
    scenarios, holdout = make_training_data()

    latest_training = max(s.timestamp for samples in scenarios.values() for s in samples)
    assert min(s.timestamp for s in holdout) > latest_training


@pytest.mark.behavioural
def test_process_summaries_threaded_through_without_error(storage_engine: Engine, tmp_path: Path):
    """Phase 5: an optional process summary can be merged into the feature
    matrix without breaking training -- the wiring the ablation's "v2" arm
    depends on. Omitting it (the default) is exercised by every other test
    in this file."""
    import pandas as pd

    from aiperfopt.features.process_features import PROCESS_SUMMARY_COLUMNS

    fconfig = _fast_feature_config()
    fsettings = _fast_forecasting_settings(tmp_path / "models")

    scenarios, holdout_samples = make_training_data()

    def _fake_process_summary(samples) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": [s.timestamp for s in samples],
            **{col: 1.0 for col in PROCESS_SUMMARY_COLUMNS},
        })

    process_summaries = {name: _fake_process_summary(samples) for name, samples in scenarios.items()}
    holdout_process_summary = _fake_process_summary(holdout_samples)

    summary = train_and_evaluate_all(
        storage_engine, fconfig, fsettings, model_version="test-v2",
        target_metrics=["cpu_percent"], scenarios=scenarios, holdout_samples=holdout_samples, holdout_dataset_tag=HOLDOUT_TAG,
        process_summaries=process_summaries, holdout_process_summary=holdout_process_summary,
        compute_feature_importance=True,
    )

    assert len(summary.outcomes) == 2  # threading a process summary through must not break anything
    # Whether LightGBM beats the baselines on this seed/duration is not
    # this test's concern (see test_lightgbm_artifact_saved_only_when_
    # selected's identical handling) -- only check the SHAP importance
    # contract when a lightgbm model actually exists to compute it on.
    for outcome in summary.outcomes:
        if outcome.selected_forecaster != "lightgbm":
            continue
        assert outcome.feature_importance  # SHAP importance populated when opted in
        # A process-summary column should appear among the features a
        # lightgbm model trained on the merged matrix actually used.
        assert any(name.startswith("proc_") for name in outcome.feature_importance)


@pytest.mark.behavioural
def test_feature_importance_absent_when_not_requested(storage_engine: Engine, tmp_path: Path):
    fconfig = _fast_feature_config()
    fsettings = _fast_forecasting_settings(tmp_path / "models")

    summary = train_and_evaluate_all(
        storage_engine, fconfig, fsettings, model_version="test-v1",
        target_metrics=["cpu_percent"], **_data_kwargs(),
    )
    assert all(o.feature_importance is None for o in summary.outcomes)
