from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine

from aiperfopt.anomaly.detector import AnomalyDetector
from aiperfopt.anomaly.notifications import emit_anomaly_notification_if_significant
from aiperfopt.anomaly.persistence import (
    anomaly_artifact_exists,
    load_anomaly_detector,
    save_anomaly_detector,
)
from aiperfopt.common.exceptions import AnomalyDetectionError
from aiperfopt.storage import get_log_events_range


def _trained_detector() -> AnomalyDetector:
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.normal(0, 1, 50)})
    detector = AnomalyDetector(min_training_rows=20)
    detector.fit(data)
    return detector


def test_save_and_load_round_trip(tmp_path: Path):
    detector = _trained_detector()
    save_anomaly_detector(detector, tmp_path, "v1")
    assert anomaly_artifact_exists(tmp_path, "v1") is True

    loaded = load_anomaly_detector(tmp_path, "v1")
    row = pd.Series({"a": 0.0, "b": 0.0})
    assert loaded.score(row) == pytest.approx(detector.score(row))


def test_save_untrained_raises(tmp_path: Path):
    with pytest.raises(AnomalyDetectionError):
        save_anomaly_detector(AnomalyDetector(), tmp_path, "v1")


def test_load_missing_artifact_raises(tmp_path: Path):
    with pytest.raises(AnomalyDetectionError):
        load_anomaly_detector(tmp_path, "does_not_exist")


def test_notification_emitted_when_score_meets_threshold(storage_engine: Engine):
    emitted = emit_anomaly_notification_if_significant(
        storage_engine, anomaly_score=0.9, threshold=0.7, timestamp=100.0
    )
    assert emitted is True

    events = get_log_events_range(storage_engine)
    assert len(events) == 1
    assert events[0].logger == "aiperfopt.anomaly"
    assert events[0].level == "WARNING"
    assert events[0].context["anomaly_score"] == 0.9


def test_no_notification_when_score_below_threshold(storage_engine: Engine):
    emitted = emit_anomaly_notification_if_significant(
        storage_engine, anomaly_score=0.3, threshold=0.7, timestamp=100.0
    )
    assert emitted is False
    assert get_log_events_range(storage_engine) == []


def test_notification_at_exact_threshold_is_emitted(storage_engine: Engine):
    emitted = emit_anomaly_notification_if_significant(
        storage_engine, anomaly_score=0.7, threshold=0.7, timestamp=100.0
    )
    assert emitted is True
