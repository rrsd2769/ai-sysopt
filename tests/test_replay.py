from __future__ import annotations

import math
import time
from pathlib import Path

import pytest
from sqlalchemy import Engine

from aiperfopt.common.exceptions import ReplayError
from aiperfopt.config.settings import ContentionSettings, DiagnosisSettings, ForecastingSettings
from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.contention.dataset import build_live_contention_input_row
from aiperfopt.contention.persistence import load_contention_classifier
from aiperfopt.contention.prediction import predict_contention
from aiperfopt.contention.training_pipeline import train_contention_and_anomaly_pipeline
from aiperfopt.anomaly.persistence import load_anomaly_detector
from aiperfopt.diagnosis.service import diagnose_contention_event
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.features.storage_bridge import compute_live_feature_vector
from aiperfopt.forecasting.deployment import resolve_deployed_forecasters
from aiperfopt.forecasting.training_pipeline import train_and_evaluate_all
from aiperfopt.replay import player as player_module
from aiperfopt.replay.player import is_monitoring_active, replay_session
from aiperfopt.replay.session import RecordedSession, load_session, record_session, save_session
from aiperfopt.storage import create_all_tables, create_storage_engine, insert_system_metric_samples_batch
from aiperfopt.storage.reader import get_system_metrics_range
from aiperfopt.storage.records import ProcessMetricSample, SystemMetricSample


def _flat_samples(n: int, interval: float, start: float = 0.0) -> list[SystemMetricSample]:
    return [
        SystemMetricSample(timestamp=start + i * interval, cpu_percent=20.0, mem_percent=50.0)
        for i in range(n)
    ]


def _rising_cpu_samples(n: int, interval: float = 2.0, start: float = 0.0) -> list[SystemMetricSample]:
    return [
        SystemMetricSample(
            timestamp=start + i * interval,
            cpu_percent=max(0.0, min(95.0, 5.0 + (i / n) * 90.0 + 4.0 * math.sin(i / 3.0))),
            mem_percent=50.0,
        )
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay's pacing sleeps are real-time wall-clock pacing, orthogonal to
    the correctness of what gets written (see player.py's module
    docstring) -- mocked out everywhere in this file so tests stay fast
    regardless of session span or speed."""
    monkeypatch.setattr(player_module.time, "sleep", lambda _seconds: None)


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
    dsettings = DiagnosisSettings(
        cross_resource_dominance_ratio=0.6, process_dominance_ratio=0.5,
        memory_sudden_rate_of_change_threshold=0.2, shap_top_n=5,
    )
    labeling_config = ContentionLabelingConfig(value_thresholds={"cpu_percent": 60.0}, tolerance_seconds=3.0)
    return fconfig, fsettings, csettings, dsettings, labeling_config


# --- record_session / save_session / load_session --------------------------------


def test_record_session_raises_when_no_data(storage_engine: Engine):
    with pytest.raises(ReplayError):
        record_session(storage_engine, minutes=5.0, end=1000.0)


def test_record_session_captures_actual_data_span(storage_engine: Engine):
    insert_system_metric_samples_batch(storage_engine, _flat_samples(10, interval=2.0, start=100.0))

    session = record_session(storage_engine, minutes=5.0, end=200.0)

    assert len(session.system_metrics) == 10
    assert session.start == pytest.approx(100.0)
    assert session.end == pytest.approx(118.0)
    assert session.duration_seconds == pytest.approx(18.0)


def test_save_and_load_session_round_trips(storage_engine: Engine, tmp_path: Path):
    insert_system_metric_samples_batch(storage_engine, _flat_samples(5, interval=1.0, start=0.0))
    from aiperfopt.storage.writer import insert_process_metric_samples_batch

    insert_process_metric_samples_batch(
        storage_engine,
        [ProcessMetricSample(timestamp=1.5, pid=42, name="chrome", cpu_percent=12.5)],
    )

    session = record_session(storage_engine, minutes=5.0, end=10.0)
    out_path = tmp_path / "session.json"
    save_session(session, out_path)
    loaded = load_session(out_path)

    assert loaded.start == session.start
    assert loaded.end == session.end
    assert len(loaded.system_metrics) == len(session.system_metrics)
    assert [s.timestamp for s in loaded.system_metrics] == [s.timestamp for s in session.system_metrics]
    assert len(loaded.process_metrics) == 1
    assert loaded.process_metrics[0].name == "chrome"


# --- guard: refuse while monitoring active ----------------------------------------


def test_is_monitoring_active_true_for_fresh_sample(storage_engine: Engine):
    insert_system_metric_samples_batch(storage_engine, [SystemMetricSample(
        timestamp=time.time(), cpu_percent=10.0, mem_percent=10.0,
    )])
    assert is_monitoring_active(storage_engine) is True


def test_is_monitoring_active_false_for_stale_or_missing_data(storage_engine: Engine):
    assert is_monitoring_active(storage_engine) is False

    insert_system_metric_samples_batch(storage_engine, [SystemMetricSample(
        timestamp=time.time() - 3600.0, cpu_percent=10.0, mem_percent=10.0,
    )])
    assert is_monitoring_active(storage_engine) is False


def test_replay_refuses_while_primary_engine_looks_active(storage_engine: Engine, tmp_path: Path):
    primary = storage_engine
    insert_system_metric_samples_batch(primary, [SystemMetricSample(
        timestamp=time.time(), cpu_percent=10.0, mem_percent=10.0,
    )])
    session = RecordedSession(
        recorded_at=time.time(), start=0.0, end=10.0,
        system_metrics=tuple(_flat_samples(5, interval=2.0)),
    )

    replay_engine = create_storage_engine(tmp_path / "replay.db")
    create_all_tables(replay_engine)

    with pytest.raises(ReplayError):
        replay_session(session, replay_engine, primary_engine=primary)

    assert get_system_metrics_range(replay_engine) == []


def test_replay_proceeds_when_primary_engine_is_stale(storage_engine: Engine, tmp_path: Path):
    primary = storage_engine
    insert_system_metric_samples_batch(primary, [SystemMetricSample(
        timestamp=time.time() - 7200.0, cpu_percent=10.0, mem_percent=10.0,
    )])
    session = RecordedSession(
        recorded_at=time.time(), start=0.0, end=10.0,
        system_metrics=tuple(_flat_samples(5, interval=2.0)),
    )

    replay_engine = create_storage_engine(tmp_path / "replay.db")
    create_all_tables(replay_engine)

    replay_session(session, replay_engine, primary_engine=primary)

    assert len(get_system_metrics_range(replay_engine)) == 5


# --- replay writes only to the replay database ------------------------------------


def test_replay_writes_nothing_to_primary_database(tmp_path: Path):
    primary = create_storage_engine(tmp_path / "primary.db")
    create_all_tables(primary)
    insert_system_metric_samples_batch(primary, _flat_samples(20, interval=2.0, start=0.0))

    session = record_session(primary, minutes=5.0, end=38.0)
    primary_count_before = len(get_system_metrics_range(primary))

    replay_engine = create_storage_engine(tmp_path / "replay.db")
    create_all_tables(replay_engine)
    summary = replay_session(session, replay_engine, speed=5.0)

    assert len(get_system_metrics_range(primary)) == primary_count_before
    assert summary.system_metric_count == 20
    assert len(get_system_metrics_range(replay_engine)) == 20


# --- identical feature vectors regardless of speed --------------------------------


def test_replay_produces_identical_feature_vectors_at_different_speeds(tmp_path: Path):
    fconfig, *_ = _fast_settings(tmp_path)

    source = create_storage_engine(tmp_path / "source.db")
    create_all_tables(source)
    insert_system_metric_samples_batch(source, _rising_cpu_samples(150, interval=2.0))

    session = record_session(source, minutes=1000.0, end=400.0)
    anchor_end = 100_000.0  # fixed, deterministic anchor for both runs

    engine_1x = create_storage_engine(tmp_path / "replay_1x.db")
    create_all_tables(engine_1x)
    replay_session(session, engine_1x, speed=1.0, anchor_end=anchor_end)

    engine_10x = create_storage_engine(tmp_path / "replay_10x.db")
    create_all_tables(engine_10x)
    replay_session(session, engine_10x, speed=10.0, anchor_end=anchor_end)

    vector_1x = compute_live_feature_vector(engine_1x, fconfig, as_of=anchor_end)
    vector_10x = compute_live_feature_vector(engine_10x, fconfig, as_of=anchor_end)

    assert vector_1x is not None and vector_10x is not None
    assert vector_1x.feature_values == pytest.approx(vector_10x.feature_values)


# --- replayed contention event reproduces the original prediction/diagnosis ------


def test_replayed_contention_event_matches_original_prediction_and_diagnosis(tmp_path: Path):
    fconfig, fsettings, csettings, dsettings, labeling_config = _fast_settings(tmp_path)

    source = create_storage_engine(tmp_path / "source.db")
    create_all_tables(source)
    insert_system_metric_samples_batch(source, _rising_cpu_samples(300, interval=2.0))

    forecast_summary = train_and_evaluate_all(
        source, fconfig, fsettings, model_version="v1", target_metrics=["cpu_percent"]
    )
    short_outcomes = {
        o.metric_name: o for o in forecast_summary.outcomes
        if o.horizon_label == "short" and o.selected_forecaster != "none"
    }
    train_contention_and_anomaly_pipeline(
        source, fconfig, fsettings, csettings, labeling_config, short_outcomes, model_version="v1",
    )

    forecasters = resolve_deployed_forecasters(
        source, fconfig, fsettings, "v1", horizon_label="short", target_metrics=["cpu_percent"]
    )
    classifier = load_contention_classifier(fsettings.models_dir, "v1")
    detector = load_anomaly_detector(fsettings.models_dir, "v1")

    original_as_of = 598.0
    original_recent = get_system_metrics_range(source, start=original_as_of - fconfig.min_history_seconds, end=original_as_of)
    original_feature_vector = compute_live_feature_vector(source, fconfig, as_of=original_as_of)
    assert original_feature_vector is not None
    original_row = build_live_contention_input_row(
        original_recent, original_feature_vector, forecasters, detector, fconfig
    )
    assert original_row is not None

    original_prediction = predict_contention(
        classifier, source, original_row, as_of=original_as_of, decision_threshold=0.5,
        anomaly_dampening_factor=0.3, contributing_processes_top_n=3, model_version="v1",
    )
    original_diagnosis = diagnose_contention_event(
        source, original_prediction, original_row, forecasters, csettings, dsettings, model_version="v1",
    )

    # Replay the exact same recorded data into a separate database, anchored
    # so "now" is a fixed, known offset from the original recording.
    session = record_session(source, minutes=1000.0, end=600.0)
    anchor_end = 500_000.0
    shift = anchor_end - session.end

    replay_engine = create_storage_engine(tmp_path / "replay.db")
    create_all_tables(replay_engine)
    replay_session(session, replay_engine, speed=8.0, anchor_end=anchor_end)

    replay_as_of = original_as_of + shift
    replay_recent = get_system_metrics_range(
        replay_engine, start=replay_as_of - fconfig.min_history_seconds, end=replay_as_of
    )
    replay_feature_vector = compute_live_feature_vector(replay_engine, fconfig, as_of=replay_as_of)
    assert replay_feature_vector is not None
    replay_row = build_live_contention_input_row(
        replay_recent, replay_feature_vector, forecasters, detector, fconfig
    )
    assert replay_row is not None

    assert replay_feature_vector.feature_values == pytest.approx(original_feature_vector.feature_values)

    replay_prediction = predict_contention(
        classifier, replay_engine, replay_row, as_of=replay_as_of, decision_threshold=0.5,
        anomaly_dampening_factor=0.3, contributing_processes_top_n=3, model_version="v1",
    )
    assert replay_prediction.probability == pytest.approx(original_prediction.probability)
    assert replay_prediction.triggered == original_prediction.triggered

    replay_diagnosis = diagnose_contention_event(
        replay_engine, replay_prediction, replay_row, forecasters, csettings, dsettings, model_version="v1",
    )
    if original_diagnosis is None:
        assert replay_diagnosis is None
    else:
        assert replay_diagnosis is not None
        assert replay_diagnosis.category == original_diagnosis.category
        assert replay_diagnosis.trigger_reason == original_diagnosis.trigger_reason
