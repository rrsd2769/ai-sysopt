"""Readiness must name the reason the live loop cannot predict.

The live loop returns early on two conditions -- not enough trailing
history, then no deployed model -- and previously did so silently while
the UI showed a green "Live" banner. These tests pin each state to the
condition that actually causes it, in the same order ``_tick`` checks
them.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from sqlalchemy import Engine

from aiperfopt.config import load_settings
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.readiness import ReadinessState, assess_readiness
from aiperfopt.lifecycle.active_version import write_active_version
from aiperfopt.storage import insert_system_metric_samples_batch
from aiperfopt.storage.records import SystemMetricSample

pytestmark = pytest.mark.behavioural

FEATURE_CONFIG = FeatureEngineeringConfig()


@pytest.fixture
def retraining_settings(tmp_path: Path):
    base = load_settings().retraining
    return base.model_copy(update={"active_version_file": tmp_path / "active.json"})


def _samples_spanning(seconds: float, interval: float = 1.5) -> list[SystemMetricSample]:
    now = time.time()
    count = max(1, int(seconds / interval))
    return [
        SystemMetricSample(timestamp=now - (count - i) * interval, cpu_percent=10.0, mem_percent=50.0)
        for i in range(count)
    ]


def test_no_samples_reports_no_data(storage_engine: Engine, retraining_settings) -> None:
    result = assess_readiness(storage_engine, FEATURE_CONFIG, retraining_settings)
    assert result.state is ReadinessState.NO_DATA
    assert not result.can_predict
    assert "aiperfopt collect" in result.detail


def test_partial_history_reports_collecting_with_progress(storage_engine: Engine, retraining_settings) -> None:
    insert_system_metric_samples_batch(storage_engine, _samples_spanning(30))
    result = assess_readiness(storage_engine, FEATURE_CONFIG, retraining_settings)

    assert result.state is ReadinessState.COLLECTING
    assert not result.can_predict
    assert 0 < result.history_seconds < result.history_seconds_required


def test_enough_history_but_no_model_reports_needs_training(
    storage_engine: Engine, retraining_settings
) -> None:
    insert_system_metric_samples_batch(
        storage_engine, _samples_spanning(FEATURE_CONFIG.min_history_seconds + 60)
    )
    result = assess_readiness(storage_engine, FEATURE_CONFIG, retraining_settings)

    assert result.state is ReadinessState.NEEDS_TRAINING
    assert not result.can_predict
    assert result.active_model_version is None
    assert "aiperfopt train" in result.detail


def test_history_plus_active_version_reports_ready(storage_engine: Engine, retraining_settings) -> None:
    insert_system_metric_samples_batch(
        storage_engine, _samples_spanning(FEATURE_CONFIG.min_history_seconds + 60)
    )
    # Written through the real writer rather than hand-rolled JSON, so the
    # test cannot drift from the on-disk format the reader expects.
    write_active_version(retraining_settings.active_version_file, "v1")

    result = assess_readiness(storage_engine, FEATURE_CONFIG, retraining_settings)

    assert result.state is ReadinessState.READY
    assert result.can_predict
    assert result.active_model_version == "v1"


def test_is_replay_reports_replay_state_regardless_of_underlying_data(
    storage_engine: Engine, retraining_settings
) -> None:
    # No samples at all -- the underlying assessment would be NO_DATA --
    # but is_replay must still win, per F9: the badge must be unmistakable
    # whenever replayed data is being served, not just when it's READY.
    result = assess_readiness(storage_engine, FEATURE_CONFIG, retraining_settings, is_replay=True)

    assert result.state is ReadinessState.REPLAY
    assert not result.can_predict
    assert "REPLAY" in result.detail


def test_is_replay_still_reports_can_predict_when_underlying_data_is_ready(
    storage_engine: Engine, retraining_settings
) -> None:
    insert_system_metric_samples_batch(
        storage_engine, _samples_spanning(FEATURE_CONFIG.min_history_seconds + 60)
    )
    write_active_version(retraining_settings.active_version_file, "v1")

    result = assess_readiness(storage_engine, FEATURE_CONFIG, retraining_settings, is_replay=True)

    assert result.state is ReadinessState.REPLAY
    assert result.can_predict
    assert result.active_model_version == "v1"
