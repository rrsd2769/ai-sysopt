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


@pytest.mark.behavioural
def test_save_and_load_round_trip(tmp_path: Path):
    detector = _trained_detector()
    save_anomaly_detector(detector, tmp_path, "v1")
    assert anomaly_artifact_exists(tmp_path, "v1") is True

    loaded = load_anomaly_detector(tmp_path, "v1")
    row = pd.Series({"a": 0.0, "b": 0.0})
    assert loaded.score(row) == pytest.approx(detector.score(row))


@pytest.mark.plumbing
def test_save_untrained_raises(tmp_path: Path):
    with pytest.raises(AnomalyDetectionError):
        save_anomaly_detector(AnomalyDetector(), tmp_path, "v1")


@pytest.mark.plumbing
def test_load_missing_artifact_raises(tmp_path: Path):
    with pytest.raises(AnomalyDetectionError):
        load_anomaly_detector(tmp_path, "does_not_exist")


@pytest.mark.plumbing
def test_load_pre_phase4_artifact_raises(tmp_path: Path):
    """A pre-Phase-4 artifact lacks ``feature_set_version`` entirely (same
    unpickling-absence trap as schema_version, above) -- loading it must
    raise rather than silently score a v2 matrix with v1 feature_columns,
    dropping every process column via _prepare's reindex."""
    detector = _trained_detector()
    del detector.feature_set_version
    save_anomaly_detector(detector, tmp_path, "v1")

    with pytest.raises(AnomalyDetectionError):
        load_anomaly_detector(tmp_path, "v1")


@pytest.mark.plumbing
def test_load_pre_phase3_artifact_raises(tmp_path: Path):
    """A pre-Phase-3 artifact lacks ``schema_version``/``_train_means`` entirely.

    Simulated by deleting those attributes after fitting, standing in
    for an object pickled before this class carried them (unpickling
    such an object leaves them simply absent from __dict__, same as
    this delattr does) -- loading it must raise rather than silently
    fall back to the old, divergent score()/score_batch() imputation.
    """
    detector = _trained_detector()
    del detector.schema_version
    del detector._train_means
    save_anomaly_detector(detector, tmp_path, "v1")

    with pytest.raises(AnomalyDetectionError):
        load_anomaly_detector(tmp_path, "v1")


@pytest.mark.behavioural
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


@pytest.mark.behavioural
def test_no_notification_when_score_below_threshold(storage_engine: Engine):
    emitted = emit_anomaly_notification_if_significant(
        storage_engine, anomaly_score=0.3, threshold=0.7, timestamp=100.0
    )
    assert emitted is False
    assert get_log_events_range(storage_engine) == []


@pytest.mark.behavioural
def test_notification_at_exact_threshold_is_emitted(storage_engine: Engine):
    emitted = emit_anomaly_notification_if_significant(
        storage_engine, anomaly_score=0.7, threshold=0.7, timestamp=100.0
    )
    assert emitted is True
