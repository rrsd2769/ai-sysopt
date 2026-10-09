from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine

from aiperfopt.config.settings import ContentionSettings, DiagnosisSettings
from aiperfopt.contention.model import ContentionClassifier
from aiperfopt.contention.prediction import predict_contention
from aiperfopt.insight.diagnosis import diagnose_contention_event
from aiperfopt.insight.taxonomy import DiagnosisCategory
from aiperfopt.forecasting.model import MetricForecaster
from aiperfopt.storage import (
    ProcessMetricSample,
    get_diagnoses_range,
    insert_process_metric_samples_batch,
)

pytestmark = pytest.mark.behavioural

_CONTENTION_SETTINGS = ContentionSettings(
    decision_threshold=0.5,
    anomaly_dampening_factor=0.3,
    anomaly_significant_threshold=0.7,
    contributing_processes_top_n=3,
)
_DIAGNOSIS_SETTINGS = DiagnosisSettings(
    cross_resource_dominance_ratio=0.6,
    process_dominance_ratio=0.5,
    memory_sudden_rate_of_change_threshold=0.2,
    shap_top_n=5,
)


def _cpu_classifier() -> ContentionClassifier:
    rng = np.random.default_rng(0)
    n = 200
    a = rng.uniform(0, 100, n)
    X = pd.DataFrame({"forecast_cpu_percent": a, "current_cpu_percent": a, "anomaly_score": np.zeros(n)})
    y = pd.Series((a > 50).astype(int))
    classifier = ContentionClassifier(min_training_rows=20)
    classifier.fit(X, y)
    return classifier


def _cpu_forecaster() -> MetricForecaster:
    rng = np.random.default_rng(0)
    n = 150
    X = pd.DataFrame({
        "cpu_percent_lag_30s": rng.uniform(0, 100, n),
        "cpu_percent_roll_mean_30s": rng.uniform(0, 100, n),
        "mem_percent_rate_of_change": rng.uniform(-1, 1, n),
    })
    y = X["cpu_percent_lag_30s"] * 0.8 + rng.normal(0, 1, n)
    forecaster = MetricForecaster(
        metric_name="cpu_percent", horizon_label="short", horizon_seconds=45, model_version="v1"
    )
    forecaster.fit(X, y)
    forecaster.set_confidence_from_validation(rmse=2.0, target_std=20.0)
    return forecaster


def test_no_diagnosis_when_not_triggered_and_no_anomaly(storage_engine: Engine):
    classifier = _cpu_classifier()
    row = pd.Series({"forecast_cpu_percent": 5.0, "current_cpu_percent": 5.0, "anomaly_score": 0.0})
    contention_record = predict_contention(
        classifier, storage_engine, row, as_of=100.0, decision_threshold=0.9999,
        anomaly_dampening_factor=0.0, contributing_processes_top_n=3, model_version="v1",
    )

    result = diagnose_contention_event(
        storage_engine, contention_record, pd.Series({"mem_percent_rate_of_change": 0.0}),
        {}, _CONTENTION_SETTINGS, _DIAGNOSIS_SETTINGS, model_version="v1",
    )

    assert result is None
    assert get_diagnoses_range(storage_engine) == []


def test_triggered_cpu_event_persists_single_process_diagnosis(storage_engine: Engine):
    classifier = _cpu_classifier()
    insert_process_metric_samples_batch(storage_engine, [
        ProcessMetricSample(timestamp=99.0, pid=1, name="stress_test", cpu_percent=99.0),
        ProcessMetricSample(timestamp=99.0, pid=2, name="idle_proc", cpu_percent=1.0),
    ])
    row = pd.Series({"forecast_cpu_percent": 95.0, "current_cpu_percent": 95.0, "anomaly_score": 0.0})
    contention_record = predict_contention(
        classifier, storage_engine, row, as_of=100.0, decision_threshold=0.5,
        anomaly_dampening_factor=0.3, contributing_processes_top_n=3, model_version="v1",
    )
    forecaster = _cpu_forecaster()
    feature_row = pd.Series({
        "cpu_percent_lag_30s": 90.0, "cpu_percent_roll_mean_30s": 85.0, "mem_percent_rate_of_change": 0.05,
    })

    result = diagnose_contention_event(
        storage_engine, contention_record, feature_row, {"cpu_percent": forecaster},
        _CONTENTION_SETTINGS, _DIAGNOSIS_SETTINGS, model_version="v1",
    )

    assert result is not None
    assert result.category == DiagnosisCategory.CPU_SINGLE_PROCESS.value
    assert result.is_actionable is True
    assert result.explanation_tier == "shap"  # triggered => high-confidence trigger per Section 10.5
    assert "stress_test" in result.plain_language_explanation
    assert len(result.forecast_explanations) == 1
    assert result.forecast_explanations[0]["metric_name"] == "cpu_percent"
    assert result.forecast_explanations[0]["tier"] == "shap"

    # Regression test: diagnose_contention_event() originally discarded the
    # int id returned by insert_diagnosis(), leaving result.id == None even
    # though the row was persisted — silently breaking FR-9.3's requirement
    # that a downstream Recommendation Engine (Phase 7) be able to trace
    # back to the specific diagnosis. Caught via manual smoke-testing
    # while building Phase 7, before Phase 7's own tests were written.
    assert result.id is not None

    persisted = get_diagnoses_range(storage_engine)
    assert len(persisted) == 1
    assert persisted[0].category == DiagnosisCategory.CPU_SINGLE_PROCESS.value
    assert persisted[0].id == result.id


def test_high_anomaly_with_triggered_cpu_contention_resolves_to_cpu_category(storage_engine: Engine):
    """Phase 3, F2: a triggered resource category wins over a high anomaly score.

    Same triggering row as ``test_triggered_cpu_event_persists_single_process_diagnosis``,
    but with ``anomaly_score`` significant (0.9 >= the 0.7 threshold). Before
    Phase 3 this used to short-circuit straight to UNUSUAL_PATTERN_ANOMALY,
    which is exactly the F1/F2 bug: since a pre-Phase-3 detector scored
    every real sample close to 1.0, this path was effectively the *only*
    one ever reachable in practice, and the other 7 actionable categories
    were dead. The anomaly score is still visible on the record; it's
    just no longer the category.
    """
    classifier = _cpu_classifier()
    row = pd.Series({"forecast_cpu_percent": 95.0, "current_cpu_percent": 95.0, "anomaly_score": 0.9})
    contention_record = predict_contention(
        classifier, storage_engine, row, as_of=100.0, decision_threshold=0.5,
        anomaly_dampening_factor=0.0, contributing_processes_top_n=3, model_version="v1",
    )
    assert contention_record.triggered is True

    result = diagnose_contention_event(
        storage_engine, contention_record, pd.Series({"mem_percent_rate_of_change": 0.0}),
        {}, _CONTENTION_SETTINGS, _DIAGNOSIS_SETTINGS, model_version="v1",
    )

    assert result is not None
    assert result.category == DiagnosisCategory.CPU_MULTI_PROCESS.value  # no process data supplied
    assert result.is_actionable is True
    assert result.confidence == pytest.approx(contention_record.probability)
    assert contention_record.anomaly_score == pytest.approx(0.9)  # still on the record, just not the category


def test_high_anomaly_without_triggered_contention_falls_back_to_anomaly_category(storage_engine: Engine):
    """Anomaly is the fallback when nothing else resolves: contention never triggered here."""
    classifier = _cpu_classifier()
    row = pd.Series({"forecast_cpu_percent": 5.0, "current_cpu_percent": 5.0, "anomaly_score": 0.9})
    contention_record = predict_contention(
        classifier, storage_engine, row, as_of=100.0, decision_threshold=0.5,
        anomaly_dampening_factor=0.0, contributing_processes_top_n=3, model_version="v1",
    )
    assert contention_record.triggered is False

    result = diagnose_contention_event(
        storage_engine, contention_record, pd.Series({"mem_percent_rate_of_change": 0.0}),
        {}, _CONTENTION_SETTINGS, _DIAGNOSIS_SETTINGS, model_version="v1",
    )

    assert result is not None
    assert result.category == DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY.value
    assert result.is_actionable is False
    assert result.confidence == pytest.approx(0.9)  # anomaly score, not contention probability
    assert result.forecast_explanations == []  # no per-metric forecast attribution for an anomaly


def test_explicit_request_forces_shap_tier_even_when_routine(storage_engine: Engine):
    classifier = _cpu_classifier()
    # A borderline-triggered row so this is a genuine "explicit request
    # upgrades the tier" case, not accidentally already high-confidence.
    row = pd.Series({"forecast_cpu_percent": 60.0, "current_cpu_percent": 60.0, "anomaly_score": 0.0})
    contention_record = predict_contention(
        classifier, storage_engine, row, as_of=100.0, decision_threshold=0.4,
        anomaly_dampening_factor=0.0, contributing_processes_top_n=3, model_version="v1",
    )
    forecaster = _cpu_forecaster()
    feature_row = pd.Series({
        "cpu_percent_lag_30s": 60.0, "cpu_percent_roll_mean_30s": 55.0, "mem_percent_rate_of_change": 0.0,
    })

    result = diagnose_contention_event(
        storage_engine, contention_record, feature_row, {"cpu_percent": forecaster},
        _CONTENTION_SETTINGS, _DIAGNOSIS_SETTINGS, model_version="v1", explicit_request=True,
    )

    assert result is not None
    assert result.explanation_tier == "shap"
    assert result.trigger_reason == "explicit_request"


def test_cross_resource_event_explains_every_deployed_metric(storage_engine: Engine):
    classifier = _cpu_classifier()
    row = pd.Series({"forecast_cpu_percent": 95.0, "current_cpu_percent": 95.0, "anomaly_score": 0.0})
    contention_record = predict_contention(
        classifier, storage_engine, row, as_of=100.0, decision_threshold=0.5,
        anomaly_dampening_factor=0.0, contributing_processes_top_n=3, model_version="v1",
    )
    # Force a Cross-Resource result by overriding contributing_metrics
    # directly (unit-level control the taxonomy tests already cover in
    # isolation) so this test can focus on the service-layer behavior:
    # explaining every deployed forecaster, not just one.
    contention_record.contributing_metrics = [
        {"metric": "forecast_cpu_percent", "contribution": 0.5},
        {"metric": "forecast_mem_percent", "contribution": 0.5},
    ]
    cpu_forecaster = _cpu_forecaster()
    mem_forecaster = _cpu_forecaster()  # architecture-identical stand-in; only used as a second deployed metric
    feature_row = pd.Series({
        "cpu_percent_lag_30s": 90.0, "cpu_percent_roll_mean_30s": 85.0, "mem_percent_rate_of_change": 0.05,
    })

    result = diagnose_contention_event(
        storage_engine, contention_record, feature_row,
        {"cpu_percent": cpu_forecaster, "mem_percent": mem_forecaster},
        _CONTENTION_SETTINGS, _DIAGNOSIS_SETTINGS, model_version="v1",
    )

    assert result.category == DiagnosisCategory.CROSS_RESOURCE_CONTENTION.value
    explained_metrics = {fx["metric_name"] for fx in result.forecast_explanations}
    assert explained_metrics == {"cpu_percent", "mem_percent"}
