from __future__ import annotations

import math
from pathlib import Path

import pytest
from sqlalchemy import Engine

from aiperfopt.config.settings import ContentionSettings, ForecastingSettings, RetrainingSettings
from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.features import FeatureEngineeringConfig
from aiperfopt.retraining.deployment import confirm_deployment, get_active_model_version
from aiperfopt.retraining.pipeline import (
    run_retraining_on_real_usage_data,
)
from aiperfopt.storage import get_evaluations_by_version, insert_system_metric_samples_batch
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
    rsettings = RetrainingSettings(
        active_version_file=tmp_path / "active_model_version.json",
        regression_tolerance=0.10, real_usage_holdout_fraction=0.2,
    )
    return fconfig, fsettings, csettings, labeling_config, rsettings


def _rising_cpu_samples(n: int, start: float = 0.0, interval: float = 2.0) -> list[SystemMetricSample]:
    """A gently rising, noisy CPU trend -- enough signal for LightGBM and
    enough threshold-crossings for contention labeling to have positive
    examples.
    """
    samples = []
    for i in range(n):
        t = start + i * interval
        cpu = min(95.0, 5.0 + (i / n) * 90.0 + 4.0 * math.sin(i / 3.0))
        samples.append(SystemMetricSample(timestamp=t, cpu_percent=max(0.0, cpu), mem_percent=50.0))
    return samples


@pytest.mark.behavioural
def test_run_retraining_on_real_usage_data_produces_candidate(storage_engine: Engine, tmp_path: Path):
    fconfig, fsettings, csettings, labeling_config, rsettings = _fast_settings(tmp_path)
    insert_system_metric_samples_batch(storage_engine, _rising_cpu_samples(300))

    candidate = run_retraining_on_real_usage_data(
        storage_engine, fconfig, fsettings, csettings, labeling_config, rsettings,
        new_model_version="rt-v1", target_metrics=["cpu_percent"],
    )

    assert candidate.model_version == "rt-v1"
    assert candidate.data_source == "real_usage"
    assert len(candidate.forecast_summary.outcomes) == 2  # short + medium horizon
    assert candidate.contention_summary is not None
    assert candidate.contention_summary.n_training_rows > 0
    # First-ever candidate: nothing deployed yet to compare against.
    assert candidate.comparison.recommendation == "no_active_baseline"
    assert candidate.comparison.active_version is None

    # Holdout evaluations were tagged distinctly from the synthetic case.
    rows = get_evaluations_by_version(storage_engine, "rt-v1")
    holdout_tags = {r.dataset_tag for r in rows if r.dataset_tag and r.dataset_tag.startswith("holdout_")}
    assert holdout_tags == {"holdout_real_usage"}


@pytest.mark.behavioural
def test_second_candidate_compares_against_first_deployed_version(storage_engine: Engine, tmp_path: Path):
    fconfig, fsettings, csettings, labeling_config, rsettings = _fast_settings(tmp_path)
    insert_system_metric_samples_batch(storage_engine, _rising_cpu_samples(300))

    first_candidate = run_retraining_on_real_usage_data(
        storage_engine, fconfig, fsettings, csettings, labeling_config, rsettings,
        new_model_version="rt-v1", target_metrics=["cpu_percent"],
    )
    confirm_deployment(
        storage_engine, rsettings, "rt-v1", first_candidate.comparison, first_candidate.skill_gate, confirmed=True
    )
    assert get_active_model_version(rsettings) == "rt-v1"

    # Retrain again on the same underlying data/settings -- should be a
    # comparable (not-regressed) candidate against the now-active rt-v1.
    second_candidate = run_retraining_on_real_usage_data(
        storage_engine, fconfig, fsettings, csettings, labeling_config, rsettings,
        new_model_version="rt-v2", target_metrics=["cpu_percent"],
    )

    assert second_candidate.comparison.active_version == "rt-v1"
    assert second_candidate.comparison.recommendation in {"deploy", "reject"}
    assert len(second_candidate.comparison.metric_comparisons) == 2  # short + medium
    for c in second_candidate.comparison.metric_comparisons:
        assert c.baseline_mae is not None  # rt-v1's holdout evaluations were found


@pytest.mark.plumbing
def test_full_confirm_and_active_version_persists_across_calls(storage_engine: Engine, tmp_path: Path):
    fconfig, fsettings, csettings, labeling_config, rsettings = _fast_settings(tmp_path)
    insert_system_metric_samples_batch(storage_engine, _rising_cpu_samples(300))

    candidate = run_retraining_on_real_usage_data(
        storage_engine, fconfig, fsettings, csettings, labeling_config, rsettings,
        new_model_version="rt-v1", target_metrics=["cpu_percent"],
    )
    record = confirm_deployment(
        storage_engine, rsettings, "rt-v1", candidate.comparison, candidate.skill_gate, confirmed=True
    )

    assert record.decision == "deployed"
    # A fresh read of the pointer file (simulating a separate process/request)
    # sees the same active version.
    assert get_active_model_version(rsettings) == "rt-v1"
