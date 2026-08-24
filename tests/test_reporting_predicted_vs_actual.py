from __future__ import annotations

import pytest
from sqlalchemy import Engine

from aiperfopt.reporting.predicted_vs_actual import join_predicted_vs_actual
from aiperfopt.storage import insert_forecast_prediction, insert_system_metric_samples_batch
from aiperfopt.storage.records import ForecastPredictionRecord, SystemMetricSample


def _sample(ts: float, cpu: float) -> SystemMetricSample:
    return SystemMetricSample(timestamp=ts, cpu_percent=cpu, mem_percent=50.0)


def test_rejects_unknown_metric_name(storage_engine: Engine):
    with pytest.raises(ValueError):
        join_predicted_vs_actual(storage_engine, "not_a_real_metric")


def test_empty_when_no_predictions(storage_engine: Engine):
    assert join_predicted_vs_actual(storage_engine, "cpu_percent") == []


def test_matches_nearest_actual_for_target_and_current_value(storage_engine: Engine):
    samples = [_sample(1000.0 + i, cpu=float(50 + i)) for i in range(20)]
    insert_system_metric_samples_batch(storage_engine, samples)

    insert_forecast_prediction(
        storage_engine,
        ForecastPredictionRecord(
            generated_at=1000.2, target_timestamp=1010.1, metric_name="cpu_percent",
            horizon_label="short", horizon_seconds=10, predicted_value=65.0,
            model_version="v1", confidence=0.7,
        ),
    )
    points = join_predicted_vs_actual(storage_engine, "cpu_percent")
    assert len(points) == 1
    p = points[0]
    assert p.actual_value == 60.0  # sample at ts=1010
    assert p.current_value == 50.0  # sample at ts=1000
    assert p.error == pytest.approx(5.0)


def test_no_nearby_actual_leaves_fields_none(storage_engine: Engine):
    insert_forecast_prediction(
        storage_engine,
        ForecastPredictionRecord(
            generated_at=1000.0, target_timestamp=1_000_000.0, metric_name="cpu_percent",
            horizon_label="short", horizon_seconds=10, predicted_value=42.0,
            model_version="v1", confidence=0.5,
        ),
    )
    points = join_predicted_vs_actual(storage_engine, "cpu_percent", end=2_000_000.0)
    assert len(points) == 1
    assert points[0].actual_value is None
    assert points[0].error is None


def test_horizon_label_filter(storage_engine: Engine):
    for horizon in ("short", "medium"):
        insert_forecast_prediction(
            storage_engine,
            ForecastPredictionRecord(
                generated_at=1000.0, target_timestamp=1010.0 if horizon == "short" else 1200.0,
                metric_name="cpu_percent", horizon_label=horizon, horizon_seconds=10,
                predicted_value=50.0, model_version="v1", confidence=0.5,
            ),
        )
    points = join_predicted_vs_actual(storage_engine, "cpu_percent", horizon_label="medium")
    assert len(points) == 1
    assert points[0].horizon_label == "medium"


def test_results_sorted_by_target_timestamp(storage_engine: Engine):
    for target in (1050.0, 1010.0, 1030.0):
        insert_forecast_prediction(
            storage_engine,
            ForecastPredictionRecord(
                generated_at=1000.0, target_timestamp=target, metric_name="cpu_percent",
                horizon_label="short", horizon_seconds=10, predicted_value=50.0,
                model_version="v1", confidence=0.5,
            ),
        )
    points = join_predicted_vs_actual(storage_engine, "cpu_percent")
    assert [p.target_timestamp for p in points] == [1010.0, 1030.0, 1050.0]
