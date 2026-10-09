from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aiperfopt.common.exceptions import ForecastingError
from aiperfopt.forecasting.model import MetricForecaster
from aiperfopt.forecasting.persistence import artifact_exists, load_forecaster, save_forecaster


def _trained_model(min_rows: int = 10) -> MetricForecaster:
    import numpy as np

    rng = np.random.default_rng(1)
    X = pd.DataFrame({
        "a": rng.uniform(0, 10, 40), "b": rng.uniform(0, 10, 40), "current_value": 0.0,
    })
    y = X["a"] * 2 + X["b"]
    model = MetricForecaster("cpu_percent", "short", 45, "test-v1", min_training_rows=min_rows)
    model.fit(X, y)
    model.set_confidence_from_validation(rmse=1.0, target_std=5.0)
    return model


@pytest.mark.plumbing
def test_save_creates_expected_files(tmp_path: Path):
    model = _trained_model()
    target_dir = save_forecaster(model, tmp_path)
    assert (target_dir / "model.txt").exists()
    assert (target_dir / "metadata.json").exists()


@pytest.mark.plumbing
def test_artifact_exists_true_after_save_false_before(tmp_path: Path):
    assert artifact_exists(tmp_path, "cpu_percent", "short", "test-v1") is False
    save_forecaster(_trained_model(), tmp_path)
    assert artifact_exists(tmp_path, "cpu_percent", "short", "test-v1") is True


@pytest.mark.behavioural
def test_load_round_trip_predicts_identically(tmp_path: Path):
    model = _trained_model()
    save_forecaster(model, tmp_path)
    loaded = load_forecaster(tmp_path, "cpu_percent", "short", "test-v1")

    row = pd.Series({"a": 3.0, "b": 4.0, "current_value": 0.0})
    original = model.predict(row)
    restored = loaded.predict(row)

    assert original.predicted_value == pytest.approx(restored.predicted_value)
    assert original.confidence == restored.confidence
    assert loaded.feature_columns == model.feature_columns


@pytest.mark.plumbing
def test_save_untrained_model_raises(tmp_path: Path):
    model = MetricForecaster("cpu_percent", "short", 45, "test-v1")
    with pytest.raises(ForecastingError):
        save_forecaster(model, tmp_path)


@pytest.mark.plumbing
def test_load_missing_artifact_raises(tmp_path: Path):
    with pytest.raises(ForecastingError):
        load_forecaster(tmp_path, "cpu_percent", "short", "does_not_exist")


@pytest.mark.plumbing
def test_load_v1_feature_set_artifact_against_v2_raises(tmp_path: Path):
    """A forecaster whose metadata.json predates FEATURE_SET_VERSION="v2"
    (the process columns) must be rejected -- loading it silently
    would mean predict()'s reindex(self.feature_columns) drops every
    process column a v2 live feature vector now offers."""
    import json

    model = _trained_model()
    target_dir = save_forecaster(model, tmp_path)

    metadata_path = target_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["feature_set_version"]
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ForecastingError):
        load_forecaster(tmp_path, "cpu_percent", "short", "test-v1")


@pytest.mark.plumbing
def test_different_horizons_and_metrics_do_not_collide(tmp_path: Path):
    short_model = _trained_model()
    short_model.horizon_label = "short"
    save_forecaster(short_model, tmp_path)

    medium_model = _trained_model()
    medium_model.horizon_label = "medium"
    save_forecaster(medium_model, tmp_path)

    assert artifact_exists(tmp_path, "cpu_percent", "short", "test-v1")
    assert artifact_exists(tmp_path, "cpu_percent", "medium", "test-v1")
