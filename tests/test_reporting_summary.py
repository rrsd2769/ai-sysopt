from __future__ import annotations

import time

import pytest
from sqlalchemy import Engine

from aiperfopt.diagnosis.taxonomy import DiagnosisCategory
from aiperfopt.reporting.summary import (
    ContentionSummary,
    build_contention_summary,
    build_diagnosis_summary,
    build_prediction_accuracy_summary,
    build_recommendation_summary,
    build_report,
    build_system_metrics_summary,
    resolve_report_window,
)
from aiperfopt.storage import (
    insert_contention_prediction,
    insert_diagnosis,
    insert_forecast_prediction,
    insert_recommendation,
    insert_system_metric_samples_batch,
)
from aiperfopt.storage.records import (
    ContentionPredictionRecord,
    DiagnosisRecord,
    ForecastPredictionRecord,
    RecommendationRecord,
    SystemMetricSample,
)


def _make_sample(ts: float, cpu: float, mem: float = 50.0) -> SystemMetricSample:
    return SystemMetricSample(timestamp=ts, cpu_percent=cpu, mem_percent=mem, load_avg_1=1.0)


# --- resolve_report_window ------------------------------------------------------------


@pytest.mark.behavioural
def test_explicit_start_end_always_wins():
    start, end = resolve_report_window(period="daily", start=100.0, end=200.0)
    assert (start, end) == (100.0, 200.0)


@pytest.mark.behavioural
def test_explicit_start_must_precede_end():
    with pytest.raises(ValueError):
        resolve_report_window(period=None, start=200.0, end=100.0)


@pytest.mark.behavioural
def test_daily_period_defaults_to_last_24h_ending_now():
    before = time.time()
    start, end = resolve_report_window(period="daily", start=None, end=None)
    after = time.time()
    assert before <= end <= after
    assert abs((end - start) - 24 * 3600.0) < 1.0


@pytest.mark.behavioural
def test_weekly_period():
    start, end = resolve_report_window(period="weekly", start=None, end=1_000_000.0)
    assert end == 1_000_000.0
    assert start == 1_000_000.0 - 7 * 24 * 3600.0


@pytest.mark.behavioural
def test_unknown_period_without_explicit_window_raises():
    with pytest.raises(ValueError):
        resolve_report_window(period="monthly", start=None, end=None)


# --- system metrics summary ------------------------------------------------------------


@pytest.mark.behavioural
def test_system_metrics_summary_computes_avg_min_max(storage_engine: Engine):
    samples = [_make_sample(1000.0 + i, cpu=float(i * 10)) for i in range(5)]  # 0,10,20,30,40
    insert_system_metric_samples_batch(storage_engine, samples)

    summaries = build_system_metrics_summary(storage_engine, 1000.0, 1004.0, metric_names=("cpu_percent",))
    assert len(summaries) == 1
    s = summaries[0]
    assert s.metric_name == "cpu_percent"
    assert s.minimum == 0.0
    assert s.maximum == 40.0
    assert s.avg == 20.0
    assert s.sample_count == 5


@pytest.mark.behavioural
def test_system_metrics_summary_with_no_samples_is_all_none(storage_engine: Engine):
    summaries = build_system_metrics_summary(storage_engine, 0.0, 100.0, metric_names=("cpu_percent",))
    assert summaries[0].avg is None
    assert summaries[0].sample_count == 0


@pytest.mark.behavioural
def test_system_metrics_summary_skips_none_values(storage_engine: Engine):
    # swap_percent defaults to None on SystemMetricSample unless set.
    samples = [_make_sample(1000.0, cpu=10.0), _make_sample(1001.0, cpu=20.0)]
    insert_system_metric_samples_batch(storage_engine, samples)
    summaries = build_system_metrics_summary(storage_engine, 1000.0, 1001.0, metric_names=("swap_percent",))
    assert summaries[0].sample_count == 0
    assert summaries[0].avg is None


# --- prediction accuracy summary ------------------------------------------------------------


@pytest.mark.behavioural
def test_prediction_accuracy_summary_computes_mae_and_directional(storage_engine: Engine):
    base = 1000.0
    samples = [_make_sample(base + i * 2, cpu=float(50 + i)) for i in range(20)]
    insert_system_metric_samples_batch(storage_engine, samples)

    # Two predictions: at generation the "current" cpu is ~50 (near t=base),
    # predicting a rise; actual values do rise, matching direction.
    insert_forecast_prediction(
        storage_engine,
        ForecastPredictionRecord(
            generated_at=base, target_timestamp=base + 10.0, metric_name="cpu_percent",
            horizon_label="short", horizon_seconds=10, predicted_value=58.0,
            model_version="v1", confidence=0.8,
        ),
    )
    insert_forecast_prediction(
        storage_engine,
        ForecastPredictionRecord(
            generated_at=base + 2.0, target_timestamp=base + 12.0, metric_name="cpu_percent",
            horizon_label="short", horizon_seconds=10, predicted_value=59.0,
            model_version="v1", confidence=0.8,
        ),
    )

    summaries = build_prediction_accuracy_summary(storage_engine, base, base + 40.0)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.metric_name == "cpu_percent"
    assert s.horizon_label == "short"
    assert s.matched_sample_count == 2
    assert s.prediction_count == 2
    assert s.mae is not None and s.mae >= 0.0
    assert s.directional_accuracy == 1.0  # both predicted a rise, both actually rose


@pytest.mark.behavioural
def test_prediction_accuracy_summary_empty_when_no_predictions(storage_engine: Engine):
    summaries = build_prediction_accuracy_summary(storage_engine, 0.0, 1000.0)
    assert summaries == []


@pytest.mark.behavioural
def test_prediction_accuracy_summary_unmatched_predictions_have_none_metrics(storage_engine: Engine):
    # Target far in the future -- nothing observed nearby.
    insert_forecast_prediction(
        storage_engine,
        ForecastPredictionRecord(
            generated_at=1000.0, target_timestamp=1_000_000.0, metric_name="cpu_percent",
            horizon_label="short", horizon_seconds=10, predicted_value=42.0,
            model_version="v1", confidence=0.5,
        ),
    )
    summaries = build_prediction_accuracy_summary(storage_engine, 0.0, 2_000_000.0)
    assert len(summaries) == 1
    assert summaries[0].mae is None
    assert summaries[0].matched_sample_count == 0
    assert summaries[0].prediction_count == 1


@pytest.mark.behavioural
def test_prediction_accuracy_summary_separates_by_horizon(storage_engine: Engine):
    samples = [_make_sample(1000.0 + i, cpu=50.0) for i in range(30)]
    insert_system_metric_samples_batch(storage_engine, samples)
    for horizon in ("short", "medium"):
        insert_forecast_prediction(
            storage_engine,
            ForecastPredictionRecord(
                generated_at=1000.0, target_timestamp=1010.0, metric_name="cpu_percent",
                horizon_label=horizon, horizon_seconds=10, predicted_value=51.0,
                model_version="v1", confidence=0.5,
            ),
        )
    summaries = build_prediction_accuracy_summary(storage_engine, 1000.0, 1030.0)
    horizons = {s.horizon_label for s in summaries}
    assert horizons == {"short", "medium"}


# --- contention summary ------------------------------------------------------------


@pytest.mark.behavioural
def test_contention_summary_counts_triggered_and_avg_probability(storage_engine: Engine):
    insert_contention_prediction(
        storage_engine,
        ContentionPredictionRecord(
            generated_at=1000.0, probability=0.8, threshold_used=0.6, triggered=True, model_version="v1",
        ),
    )
    insert_contention_prediction(
        storage_engine,
        ContentionPredictionRecord(
            generated_at=1001.0, probability=0.2, threshold_used=0.6, triggered=False, model_version="v1",
        ),
    )
    summary = build_contention_summary(storage_engine, 1000.0, 1002.0)
    assert summary.prediction_count == 2
    assert summary.triggered_count == 1
    assert summary.avg_probability == pytest.approx(0.5)


@pytest.mark.behavioural
def test_contention_summary_empty(storage_engine: Engine):
    summary = build_contention_summary(storage_engine, 0.0, 100.0)
    assert summary == ContentionSummary(0, 0, None)


# --- diagnosis summary ------------------------------------------------------------


@pytest.mark.behavioural
def test_diagnosis_summary_zero_fills_every_taxonomy_category(storage_engine: Engine):
    summary = build_diagnosis_summary(storage_engine, 0.0, 100.0)
    assert summary.total_count == 0
    assert set(summary.counts_by_category) == {c.value for c in DiagnosisCategory}
    assert all(v == 0 for v in summary.counts_by_category.values())


@pytest.mark.behavioural
def test_diagnosis_summary_counts_and_actionable(storage_engine: Engine):
    insert_diagnosis(
        storage_engine,
        DiagnosisRecord(
            generated_at=1000.0, category=DiagnosisCategory.CPU_SINGLE_PROCESS.value,
            is_actionable=True, confidence=0.9, explanation_tier="shap",
            trigger_reason="high_confidence_contention", technical_explanation="x",
            plain_language_explanation="y", model_version="v1",
        ),
    )
    insert_diagnosis(
        storage_engine,
        DiagnosisRecord(
            generated_at=1001.0, category=DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY.value,
            is_actionable=False, confidence=0.5, explanation_tier="lightweight",
            trigger_reason="anomaly", technical_explanation="x",
            plain_language_explanation="y", model_version="v1",
        ),
    )
    summary = build_diagnosis_summary(storage_engine, 1000.0, 1002.0)
    assert summary.total_count == 2
    assert summary.actionable_count == 1
    assert summary.counts_by_category[DiagnosisCategory.CPU_SINGLE_PROCESS.value] == 1
    assert summary.counts_by_category[DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY.value] == 1
    assert summary.avg_confidence == pytest.approx(0.7)


# --- recommendation summary ------------------------------------------------------------


@pytest.mark.behavioural
def test_recommendation_summary_counts_and_actions(storage_engine: Engine):
    insert_recommendation(
        storage_engine,
        RecommendationRecord(
            generated_at=1000.0, diagnosis_id=1, diagnosis_category="cpu_bound_single_process",
            actions=[{"title": "a", "description": "b", "rationale": "c"}, {"title": "d", "description": "e", "rationale": "f"}],
            catalog_version="v1",
        ),
    )
    summary = build_recommendation_summary(storage_engine, 1000.0, 1001.0)
    assert summary.total_count == 1
    assert summary.total_actions == 2
    assert summary.counts_by_diagnosis_category == {"cpu_bound_single_process": 1}


@pytest.mark.behavioural
def test_recommendation_summary_empty(storage_engine: Engine):
    summary = build_recommendation_summary(storage_engine, 0.0, 100.0)
    assert summary.total_count == 0
    assert summary.total_actions == 0
    assert summary.counts_by_diagnosis_category == {}


# --- build_report (top-level) ------------------------------------------------------------


@pytest.mark.behavioural
def test_build_report_combines_all_sub_summaries(storage_engine: Engine):
    samples = [_make_sample(1000.0 + i, cpu=50.0) for i in range(10)]
    insert_system_metric_samples_batch(storage_engine, samples)
    insert_diagnosis(
        storage_engine,
        DiagnosisRecord(
            generated_at=1000.0, category=DiagnosisCategory.CPU_SINGLE_PROCESS.value,
            is_actionable=True, confidence=0.9, explanation_tier="shap",
            trigger_reason="high_confidence_contention", technical_explanation="x",
            plain_language_explanation="y", model_version="v1",
        ),
    )

    report = build_report(storage_engine, start=1000.0, end=1010.0)
    assert report.start == 1000.0
    assert report.end == 1010.0
    assert report.generated_at > 0
    assert any(s.metric_name == "cpu_percent" for s in report.system_metrics)
    assert report.diagnosis.total_count == 1
    assert report.prediction_accuracy == []
    assert report.contention.prediction_count == 0
    assert report.recommendation.total_count == 0


@pytest.mark.plumbing
def test_build_report_requires_period_or_explicit_window(storage_engine: Engine):
    with pytest.raises(ValueError):
        build_report(storage_engine)
