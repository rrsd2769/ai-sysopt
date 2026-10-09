from __future__ import annotations

import pytest
from sqlalchemy import Engine

from aiperfopt.common.exceptions import RetrainingError
from aiperfopt.lifecycle.training_data import (
    REAL_USAGE_SCENARIO_NAME,
    build_real_usage_training_data,
)
from aiperfopt.storage import insert_system_metric_samples_batch
from aiperfopt.storage.records import SystemMetricSample


def _make_samples(n: int, start: float = 0.0, interval: float = 2.0) -> list[SystemMetricSample]:
    return [
        SystemMetricSample(timestamp=start + i * interval, cpu_percent=float(i % 100), mem_percent=50.0)
        for i in range(n)
    ]


@pytest.mark.behavioural
def test_build_real_usage_training_data_splits_chronologically(storage_engine: Engine):
    samples = _make_samples(100)
    insert_system_metric_samples_batch(storage_engine, samples)

    scenarios, holdout_samples = build_real_usage_training_data(
        storage_engine, holdout_fraction=0.2, min_total_rows=1
    )

    assert list(scenarios.keys()) == [REAL_USAGE_SCENARIO_NAME]
    training_samples = scenarios[REAL_USAGE_SCENARIO_NAME]
    assert len(training_samples) == 80
    assert len(holdout_samples) == 20

    # Chronological, non-overlapping, holdout strictly after training (ADL-10).
    assert training_samples[-1].timestamp < holdout_samples[0].timestamp
    assert [s.timestamp for s in training_samples] == sorted(s.timestamp for s in training_samples)
    assert [s.timestamp for s in holdout_samples] == sorted(s.timestamp for s in holdout_samples)


@pytest.mark.plumbing
def test_build_real_usage_training_data_raises_below_min_rows(storage_engine: Engine):
    insert_system_metric_samples_batch(storage_engine, _make_samples(5))

    with pytest.raises(RetrainingError):
        build_real_usage_training_data(storage_engine, min_total_rows=50)


@pytest.mark.plumbing
def test_build_real_usage_training_data_raises_on_empty_split(storage_engine: Engine):
    # A single row can't be split into a non-empty training set and a
    # non-empty holdout set at any fraction.
    insert_system_metric_samples_batch(storage_engine, _make_samples(1))

    with pytest.raises(RetrainingError):
        build_real_usage_training_data(storage_engine, holdout_fraction=0.2, min_total_rows=1)


@pytest.mark.behavioural
def test_build_real_usage_training_data_respects_start_end_bounds(storage_engine: Engine):
    insert_system_metric_samples_batch(storage_engine, _make_samples(100, start=0.0, interval=1.0))

    scenarios, holdout_samples = build_real_usage_training_data(
        storage_engine, start=10.0, end=30.0, holdout_fraction=0.5, min_total_rows=1
    )
    all_samples = scenarios[REAL_USAGE_SCENARIO_NAME] + holdout_samples
    assert all(10.0 <= s.timestamp <= 30.0 for s in all_samples)
