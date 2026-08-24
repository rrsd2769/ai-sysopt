from __future__ import annotations

from aiperfopt.diagnosis.explainer import ForecastExplanation, FeatureAttribution
from aiperfopt.diagnosis.taxonomy import DiagnosisCategory
from aiperfopt.diagnosis.text import build_plain_language_explanation, build_technical_explanation
from aiperfopt.storage.records import ContentionPredictionRecord


def _record(**overrides) -> ContentionPredictionRecord:
    defaults = dict(
        generated_at=100.0, probability=0.95, threshold_used=0.75, triggered=True,
        model_version="v1",
        contributing_metrics=[{"metric": "forecast_cpu_percent", "contribution": 3.2}],
        contributing_processes=[
            {"pid": 1, "name": "stress_test", "value": 99.0},
            {"pid": 2, "name": "idle_proc", "value": 1.0},
        ],
        anomaly_score=0.05,
    )
    defaults.update(overrides)
    return ContentionPredictionRecord(**defaults)


def test_cpu_single_process_names_only_the_dominant_process():
    """Regression test for a real bug caught during Phase 6 development:
    the single-process template originally named every returned process
    (e.g. "stress_test and idle_proc is using..."), contradicting the
    category's own claim that one process dominates."""
    record = _record()
    text = build_plain_language_explanation(DiagnosisCategory.CPU_SINGLE_PROCESS, record)
    assert "stress_test" in text
    assert "idle_proc" not in text


def test_cpu_single_process_with_no_process_data_uses_generic_singular_subject():
    record = _record(contributing_processes=None)
    text = build_plain_language_explanation(DiagnosisCategory.CPU_SINGLE_PROCESS, record)
    assert "One application" in text


def test_cpu_multi_process_names_multiple_processes():
    record = _record()
    text = build_plain_language_explanation(DiagnosisCategory.CPU_MULTI_PROCESS, record)
    assert "stress_test" in text
    assert "idle_proc" in text


def test_every_actionable_category_produces_nonempty_distinct_text():
    record = _record()
    seen = set()
    for category in DiagnosisCategory:
        text = build_plain_language_explanation(category, record)
        assert len(text) > 20
        assert text not in seen, f"{category} produced duplicate text"
        seen.add(text)


def test_anomaly_category_text_does_not_recommend_action():
    record = _record()
    text = build_plain_language_explanation(DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY, record)
    # ADL-36: informational only. The plain-language text shouldn't read
    # as an actionable recommendation the way the other categories do.
    assert "no action is being recommended" in text.lower() or "isn't a specific problem" in text.lower()


def test_confidence_word_scales_with_probability():
    high = build_plain_language_explanation(DiagnosisCategory.CPU_SINGLE_PROCESS, _record(probability=0.95))
    low = build_plain_language_explanation(DiagnosisCategory.CPU_SINGLE_PROCESS, _record(probability=0.76))
    assert "high confidence" in high
    assert "moderate confidence" in low


def test_technical_explanation_includes_numeric_confidence_and_tier():
    record = _record()
    fx = [ForecastExplanation(
        metric_name="cpu_percent", forecaster_kind="lightgbm", tier="shap",
        top_features=[FeatureAttribution("cpu_percent_lag_30s", 12.5)],
    )]
    text = build_technical_explanation(
        DiagnosisCategory.CPU_SINGLE_PROCESS, record, fx, "shap", "high_confidence_contention"
    )
    assert "0.950" in text  # numeric probability, not a qualitative bucket
    assert "shap" in text
    assert "cpu_percent_lag_30s" in text
    assert "stress_test" in text


def test_technical_explanation_includes_baseline_note_for_not_applicable_tier():
    record = _record()
    fx = [ForecastExplanation(
        metric_name="cpu_percent", forecaster_kind="NaivePersistenceForecaster", tier="not_applicable",
        note="'cpu_percent' is currently forecast by the NaivePersistenceForecaster baseline...",
    )]
    text = build_technical_explanation(
        DiagnosisCategory.CPU_SINGLE_PROCESS, record, fx, "lightweight", "routine"
    )
    assert "baseline" in text
