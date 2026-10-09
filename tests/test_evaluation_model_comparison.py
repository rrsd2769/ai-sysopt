from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine

from aiperfopt.bootstrap import AppContext
from aiperfopt.config.settings import ForecastingSettings
from aiperfopt.evaluation.model_comparison import CANDIDATE_NAMES, run_model_comparison
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.storage import insert_system_metric_samples_batch
from tests.factories import make_training_data


@pytest.mark.behavioural
def test_model_comparison_writes_outputs_and_picks_one_of_the_three_candidates(
    storage_engine: Engine, tmp_path: Path
):
    scenarios, holdout = make_training_data(n_train=300, n_holdout=100)
    insert_system_metric_samples_batch(storage_engine, scenarios["train"] + holdout)
    settings = ForecastingSettings(
        short_horizon_seconds=(10, 20), medium_horizon_seconds=(30, 60),
        walk_forward_splits=2, min_training_rows=15, models_dir=tmp_path / "models",
        directional_stability_threshold=0.02, baseline_improvement_margin=0.05,
    )
    ctx = AppContext(settings=type("S", (), {"forecasting": settings})(), storage_engine=storage_engine)
    feature_config = FeatureEngineeringConfig(lookback_windows_seconds=(10, 30), tolerance_seconds=3.0)

    result = run_model_comparison(ctx, tmp_path / "out", feature_config=feature_config)

    assert result.winner in CANDIDATE_NAMES
    assert set(result.skill) == set(CANDIDATE_NAMES)
    assert {p.name for p in result.files} == {
        "model_comparison.csv", "model_comparison_cv.csv", "model_comparison.txt", "model_comparison.png",
    }
    assert all(p.exists() for p in result.files)
    assert f"SELECTED: {result.winner}" in (tmp_path / "out" / "model_comparison.txt").read_text()
