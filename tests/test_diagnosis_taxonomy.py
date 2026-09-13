from __future__ import annotations

import pytest

from aiperfopt.config.settings import DiagnosisSettings
from aiperfopt.diagnosis.taxonomy import DiagnosisCategory, is_actionable, resolve_diagnosis_category
from aiperfopt.storage.records import ContentionPredictionRecord

pytestmark = pytest.mark.behavioural

_CONFIG = DiagnosisSettings(
    cross_resource_dominance_ratio=0.6,
    process_dominance_ratio=0.5,
    memory_sudden_rate_of_change_threshold=0.2,
    shap_top_n=5,
)


def _record(
    *,
    triggered: bool = True,
    probability: float = 0.9,
    contributing_metrics: list[dict] | None = None,
    contributing_processes: list[dict] | None = None,
    anomaly_score: float | None = 0.0,
) -> ContentionPredictionRecord:
    return ContentionPredictionRecord(
        generated_at=100.0,
        probability=probability,
        threshold_used=0.5,
        triggered=triggered,
        model_version="v1",
        contributing_metrics=contributing_metrics,
        contributing_processes=contributing_processes,
        anomaly_score=anomaly_score,
    )


def test_not_triggered_and_no_anomaly_resolves_to_none():
    record = _record(triggered=False, contributing_metrics=[{"metric": "forecast_cpu_percent", "contribution": 0.9}])
    result = resolve_diagnosis_category(record, None, 0.7, _CONFIG)
    assert result is None


def test_triggered_resource_category_wins_over_high_anomaly():
    """Phase 3, F2: contention/dominance resolution runs before anomaly, not after.

    A high anomaly score no longer short-circuits diagnosis when
    contention is triggered and a resource is dominant -- the resource
    category resolves as normal, and the anomaly score remains available
    on the record as a flag rather than replacing the category.
    """
    record = _record(triggered=True, anomaly_score=0.95, contributing_metrics=[
        {"metric": "forecast_cpu_percent", "contribution": 0.9}
    ])
    result = resolve_diagnosis_category(record, None, 0.7, _CONFIG)
    assert result == DiagnosisCategory.CPU_MULTI_PROCESS
    assert is_actionable(result) is True
    assert record.anomaly_score == 0.95


def test_not_triggered_with_high_anomaly_falls_back_to_anomaly_category():
    """Anomaly is the fallback when nothing else resolves -- i.e. contention never triggered."""
    record = _record(triggered=False, anomaly_score=0.95, contributing_metrics=None)
    result = resolve_diagnosis_category(record, None, 0.7, _CONFIG)
    assert result == DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY
    assert is_actionable(result) is False


def test_anomaly_below_threshold_does_not_trigger_anomaly_category():
    record = _record(triggered=True, anomaly_score=0.5, contributing_metrics=[
        {"metric": "forecast_cpu_percent", "contribution": 0.9}
    ])
    result = resolve_diagnosis_category(record, None, 0.7, _CONFIG)
    assert result != DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY


def test_dominant_cpu_with_no_process_data_defaults_to_multi_process():
    record = _record(contributing_metrics=[{"metric": "forecast_cpu_percent", "contribution": 0.9}])
    result = resolve_diagnosis_category(record, None, 0.7, _CONFIG)
    assert result == DiagnosisCategory.CPU_MULTI_PROCESS


def test_dominant_cpu_with_one_dominant_process_is_single_process():
    record = _record(
        contributing_metrics=[{"metric": "forecast_cpu_percent", "contribution": 0.9}],
        contributing_processes=[
            {"pid": 1, "name": "hog", "value": 95.0},
            {"pid": 2, "name": "idle", "value": 2.0},
        ],
    )
    result = resolve_diagnosis_category(record, None, 0.7, _CONFIG)
    assert result == DiagnosisCategory.CPU_SINGLE_PROCESS


def test_dominant_cpu_with_comparable_processes_is_multi_process():
    record = _record(
        contributing_metrics=[{"metric": "forecast_cpu_percent", "contribution": 0.9}],
        contributing_processes=[
            {"pid": 1, "name": "a", "value": 40.0},
            {"pid": 2, "name": "b", "value": 38.0},
            {"pid": 3, "name": "c", "value": 30.0},
        ],
    )
    result = resolve_diagnosis_category(record, None, 0.7, _CONFIG)
    assert result == DiagnosisCategory.CPU_MULTI_PROCESS


def test_dominant_memory_defaults_to_gradual():
    record = _record(contributing_metrics=[{"metric": "forecast_mem_percent", "contribution": 0.9}])
    result = resolve_diagnosis_category(record, 0.05, 0.7, _CONFIG)
    assert result == DiagnosisCategory.MEMORY_PRESSURE_GRADUAL


def test_dominant_memory_with_missing_rate_of_change_defaults_to_gradual():
    record = _record(contributing_metrics=[{"metric": "forecast_mem_percent", "contribution": 0.9}])
    result = resolve_diagnosis_category(record, None, 0.7, _CONFIG)
    assert result == DiagnosisCategory.MEMORY_PRESSURE_GRADUAL


def test_dominant_memory_with_sharp_rate_of_change_is_sudden():
    record = _record(contributing_metrics=[{"metric": "forecast_mem_percent", "contribution": 0.9}])
    result = resolve_diagnosis_category(record, 0.5, 0.7, _CONFIG)
    assert result == DiagnosisCategory.MEMORY_PRESSURE_SUDDEN


def test_negative_sudden_rate_of_change_also_counts():
    """A sharp *drop* is still a sudden memory event, not just a rise."""
    record = _record(contributing_metrics=[{"metric": "forecast_mem_percent", "contribution": 0.9}])
    result = resolve_diagnosis_category(record, -0.5, 0.7, _CONFIG)
    assert result == DiagnosisCategory.MEMORY_PRESSURE_SUDDEN


def test_dominant_disk_maps_to_disk_io_bottleneck():
    record = _record(contributing_metrics=[{"metric": "forecast_disk_write_bytes", "contribution": 0.9}])
    result = resolve_diagnosis_category(record, None, 0.7, _CONFIG)
    assert result == DiagnosisCategory.DISK_IO_BOTTLENECK


def test_dominant_network_maps_to_network_saturation():
    record = _record(contributing_metrics=[{"metric": "forecast_net_bytes_sent", "contribution": 0.9}])
    result = resolve_diagnosis_category(record, None, 0.7, _CONFIG)
    assert result == DiagnosisCategory.NETWORK_SATURATION


def test_no_single_dominant_resource_type_is_cross_resource():
    record = _record(contributing_metrics=[
        {"metric": "forecast_cpu_percent", "contribution": 0.5},
        {"metric": "forecast_mem_percent", "contribution": 0.5},
        {"metric": "forecast_disk_write_bytes", "contribution": 0.4},
    ])
    result = resolve_diagnosis_category(record, None, 0.7, _CONFIG)
    assert result == DiagnosisCategory.CROSS_RESOURCE_CONTENTION


def test_empty_or_unrecognized_contributing_metrics_falls_back_to_cross_resource():
    record = _record(contributing_metrics=[{"metric": "anomaly_score", "contribution": 0.9}])
    result = resolve_diagnosis_category(record, None, 0.7, _CONFIG)
    assert result == DiagnosisCategory.CROSS_RESOURCE_CONTENTION

    record_none = _record(contributing_metrics=None)
    assert resolve_diagnosis_category(record_none, None, 0.7, _CONFIG) == DiagnosisCategory.CROSS_RESOURCE_CONTENTION


def test_all_actionable_categories_except_anomaly():
    for category in DiagnosisCategory:
        expected = category != DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY
        assert is_actionable(category) is expected
