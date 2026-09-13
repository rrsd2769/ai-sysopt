from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine

from aiperfopt.common.exceptions import RetrainingError
from aiperfopt.config.settings import ContentionSettings, ForecastingSettings
from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.evaluation.ablation import feature_config_for_arm, run_ablation
from aiperfopt.storage import ProcessMetricSample, SystemMetricSample
from aiperfopt.storage.writer import insert_process_metric_sample, insert_system_metric_samples_batch


def _fast_settings(models_dir: Path):
    fconfig = FeatureEngineeringConfig(lookback_windows_seconds=(10, 30), tolerance_seconds=3.0)
    fsettings = ForecastingSettings(
        short_horizon_seconds=(10, 20), medium_horizon_seconds=(30, 60),
        walk_forward_splits=2, min_training_rows=10, models_dir=models_dir,
        directional_stability_threshold=0.02, baseline_improvement_margin=0.05,
    )
    csettings = ContentionSettings(
        decision_threshold=0.5, anomaly_dampening_factor=0.3,
        anomaly_significant_threshold=0.7, contributing_processes_top_n=3,
    )
    labeling_config = ContentionLabelingConfig(value_thresholds={"cpu_percent": 60.0}, tolerance_seconds=3.0)
    return fconfig, fsettings, csettings, labeling_config


def _seed_real_usage_data(engine: Engine, count: int = 150, interval: float = 2.0, start: float = 10_000.0) -> None:
    samples = [
        SystemMetricSample(
            timestamp=start + i * interval,
            # Oscillate with an occasional spike so labels aren't single-class.
            cpu_percent=30.0 + (40.0 if i % 25 == 0 else (i % 10)),
            mem_percent=40.0,
        )
        for i in range(count)
    ]
    insert_system_metric_samples_batch(engine, samples)

    # A handful of process polls with real CPU/launch activity, so the
    # "v2" arm actually gets non-trivial proc_* columns.
    for i in range(0, count, 3):
        insert_process_metric_sample(
            engine,
            ProcessMetricSample(
                timestamp=start + i * interval, pid=1, name="worker",
                cpu_percent=20.0 + i % 15, memory_rss=1_000_000,
                create_time=start + i * interval - 5.0,
            ),
        )


@pytest.mark.behavioural
def test_feature_config_for_arm_v1_excludes_process_columns():
    base = FeatureEngineeringConfig()
    v1 = feature_config_for_arm("v1", base)
    v2 = feature_config_for_arm("v2", base)
    assert not any(name.startswith("proc_") for name in v1.gauge_metrics)
    assert any(name.startswith("proc_") for name in v2.gauge_metrics)
    assert v2 is base


@pytest.mark.plumbing
def test_feature_config_for_arm_rejects_unknown_arm():
    with pytest.raises(ValueError):
        feature_config_for_arm("v3", FeatureEngineeringConfig())


@pytest.mark.plumbing
def test_run_ablation_raises_when_not_enough_real_data(storage_engine: Engine, tmp_path: Path):
    fconfig, fsettings, csettings, labeling_config = _fast_settings(tmp_path / "models")
    with pytest.raises(RetrainingError):
        run_ablation(
            storage_engine, fconfig, fsettings, csettings, labeling_config,
            target_metrics=["cpu_percent"], model_version_prefix="test",
            min_total_rows=1_000_000,
        )


@pytest.mark.behavioural
def test_run_ablation_produces_both_arms_with_distinct_features(storage_engine: Engine, tmp_path: Path):
    fconfig, fsettings, csettings, labeling_config = _fast_settings(tmp_path / "models")
    _seed_real_usage_data(storage_engine)

    report = run_ablation(
        storage_engine, fconfig, fsettings, csettings, labeling_config,
        target_metrics=["cpu_percent"], model_version_prefix="abltest",
        holdout_fraction=0.2, min_total_rows=10,
    )

    assert set(report.arms) == {"v1", "v2"}
    assert report.dataset_summary["training_rows"] > 0
    assert report.dataset_summary["holdout_rows"] > 0

    for arm in report.arms.values():
        assert len(arm.forecast_summary.outcomes) == 2  # short + medium horizon for cpu_percent

    # The whole point of the ablation: v2's lightgbm models (if any were
    # selected) were trained on a feature set that genuinely includes
    # process columns, v1's did not.
    v2_lightgbm = [o for o in report.arms["v2"].forecast_summary.outcomes if o.selected_forecaster == "lightgbm"]
    for outcome in v2_lightgbm:
        assert any(name.startswith("proc_") for name in (outcome.feature_importance or {}))
