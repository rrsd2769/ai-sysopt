from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import shap

from aiperfopt.contention.model import ContentionClassifier
from aiperfopt.insight.explainer import (
    EXPLANATION_TIER_LIGHTWEIGHT,
    EXPLANATION_TIER_NOT_APPLICABLE,
    EXPLANATION_TIER_SHAP,
    TRIGGER_EXPLICIT_REQUEST,
    TRIGGER_HIGH_CONFIDENCE_CONTENTION,
    TRIGGER_ROUTINE,
    TRIGGER_SIGNIFICANT_ANOMALY,
    explain_forecast,
    should_use_shap_tier,
)
from aiperfopt.forecasting.baselines import NaivePersistenceForecaster
from aiperfopt.forecasting.model import MetricForecaster

pytestmark = pytest.mark.behavioural


def _trained_forecaster(rng: np.random.Generator, n: int = 150) -> tuple[MetricForecaster, pd.Series]:
    X = pd.DataFrame({
        "cpu_percent_lag_30s": rng.uniform(0, 100, n),
        "cpu_percent_roll_mean_30s": rng.uniform(0, 100, n),
    })
    y = X["cpu_percent_lag_30s"] * 0.7 + rng.normal(0, 1, n)
    forecaster = MetricForecaster(
        metric_name="cpu_percent", horizon_label="short", horizon_seconds=45, model_version="v1"
    )
    forecaster.fit(X, y)
    feature_row = pd.Series({"cpu_percent_lag_30s": 80.0, "cpu_percent_roll_mean_30s": 60.0})
    return forecaster, feature_row


# --- should_use_shap_tier: the three triggers, priority order ---


def test_explicit_request_wins_regardless_of_other_signals():
    use_shap, reason = should_use_shap_tier(triggered=False, anomaly_score=0.0, anomaly_significant_threshold=0.7, explicit_request=True)
    assert use_shap is True
    assert reason == TRIGGER_EXPLICIT_REQUEST


def test_significant_anomaly_triggers_shap_tier():
    use_shap, reason = should_use_shap_tier(triggered=False, anomaly_score=0.8, anomaly_significant_threshold=0.7, explicit_request=False)
    assert use_shap is True
    assert reason == TRIGGER_SIGNIFICANT_ANOMALY


def test_high_confidence_contention_triggers_shap_tier():
    use_shap, reason = should_use_shap_tier(triggered=True, anomaly_score=0.0, anomaly_significant_threshold=0.7, explicit_request=False)
    assert use_shap is True
    assert reason == TRIGGER_HIGH_CONFIDENCE_CONTENTION


def test_routine_cycle_stays_lightweight():
    use_shap, reason = should_use_shap_tier(triggered=False, anomaly_score=0.1, anomaly_significant_threshold=0.7, explicit_request=False)
    assert use_shap is False
    assert reason == TRIGGER_ROUTINE


# --- explain_forecast: LightGBM (both tiers) vs. baseline (not_applicable) ---


def test_lightweight_forecast_explanation_uses_native_importance():
    rng = np.random.default_rng(0)
    forecaster, feature_row = _trained_forecaster(rng)
    result = explain_forecast("cpu_percent", forecaster, feature_row, use_shap=False, top_n=5)
    assert result.tier == EXPLANATION_TIER_LIGHTWEIGHT
    assert result.forecaster_kind == "lightgbm"
    assert len(result.top_features) > 0
    assert result.note is None


def test_shap_forecast_explanation_uses_tree_explainer():
    rng = np.random.default_rng(0)
    forecaster, feature_row = _trained_forecaster(rng)
    result = explain_forecast("cpu_percent", forecaster, feature_row, use_shap=True, top_n=5)
    assert result.tier == EXPLANATION_TIER_SHAP
    assert len(result.top_features) > 0

    # cpu_percent_lag_30s is the only feature with real predictive signal
    # (target = 0.7 * lag + noise) — it should dominate the attribution
    # regardless of which tier produced it.
    top_feature_names = {f.feature_name for f in result.top_features}
    assert "cpu_percent_lag_30s" in top_feature_names


def test_shap_forecast_explanation_matches_real_tree_shap_values():
    """The SHAP tier isn't a from-scratch reimplementation — verify it
    reproduces exactly what a direct shap.TreeExplainer call gives."""
    rng = np.random.default_rng(0)
    forecaster, feature_row = _trained_forecaster(rng)
    result = explain_forecast("cpu_percent", forecaster, feature_row, use_shap=True, top_n=5)

    x = feature_row.reindex(forecaster.feature_columns).to_frame().T.astype(float)
    explainer = shap.TreeExplainer(forecaster.booster)
    expected = dict(zip(forecaster.feature_columns, explainer.shap_values(x)[0]))

    for attribution in result.top_features:
        assert abs(attribution.value - expected[attribution.feature_name]) < 1e-9


def test_shap_and_lightweight_explanations_carry_the_full_v2_column_set():
    """Once a forecaster is trained on the v2
    feature set (system gauges + process-summary columns), both
    explanation tiers must carry every one of those columns -- not a
    subset. explain_forecast doesn't hardcode a column list; it reads
    whatever forecaster.feature_columns the trained model actually has,
    so this is a property of that genericity, verified directly rather
    than assumed."""
    rng = np.random.default_rng(2)
    n = 150
    v2_columns = {
        "cpu_percent_lag_30s": rng.uniform(0, 100, n),
        "cpu_percent_roll_mean_30s": rng.uniform(0, 100, n),
        "proc_cpu_sum_roll_mean_30s": rng.uniform(0, 100, n),
        "proc_young_cpu_sum_lag_30s": rng.uniform(0, 50, n),
    }
    X = pd.DataFrame(v2_columns)
    y = X["cpu_percent_lag_30s"] * 0.5 + X["proc_cpu_sum_roll_mean_30s"] * 0.3
    forecaster = MetricForecaster(
        metric_name="cpu_percent", horizon_label="short", horizon_seconds=45, model_version="v2"
    )
    forecaster.fit(X, y)
    feature_row = pd.Series({col: 50.0 for col in v2_columns})

    lightweight = explain_forecast("cpu_percent", forecaster, feature_row, use_shap=False, top_n=10)
    shap_result = explain_forecast("cpu_percent", forecaster, feature_row, use_shap=True, top_n=10)

    assert {f.feature_name for f in lightweight.top_features} == set(v2_columns)
    assert {f.feature_name for f in shap_result.top_features} == set(v2_columns)


def test_baseline_forecaster_explanation_is_not_applicable_with_honest_note():
    baseline = NaivePersistenceForecaster(model_version="baseline-naive_persistence")
    feature_row = pd.Series({"cpu_percent_lag_30s": 80.0})
    result = explain_forecast("cpu_percent", baseline, feature_row, use_shap=True, top_n=5)
    assert result.tier == EXPLANATION_TIER_NOT_APPLICABLE
    assert result.top_features == []
    assert result.note is not None
    assert "baseline" in result.note.lower()


# --- Contention attribution equivalence: coefficient contribution == SHAP linear ---


def test_contention_coefficient_attribution_equals_real_shap_linear_explainer():
    """Numerically verifies the claim in aiperfopt.insight.explainer's module
    docstring: ContentionClassifier.contributing_metrics() already *is* the
    exact SHAP-linear decomposition, which is why Diagnosis never recomputes
    it through a second SHAP call. If a future refactor of either
    StandardScaler usage or the coefficient formula breaks this equivalence,
    this test — not just a stale docstring — should catch it.
    """
    rng = np.random.default_rng(1)
    n = 300
    a = rng.uniform(0, 100, n)
    b = rng.uniform(0, 100, n)
    X = pd.DataFrame({"forecast_cpu_percent": a, "current_mem_percent": b})
    y = pd.Series(((a > 50) | (b > 70)).astype(int))
    classifier = ContentionClassifier(min_training_rows=20)
    classifier.fit(X, y)

    row = pd.Series({"forecast_cpu_percent": 80.0, "current_mem_percent": 90.0})
    ours = {c.metric_name: c.contribution for c in classifier.contributing_metrics(row, top_n=5)}

    x = row.reindex(classifier.feature_columns).to_numpy(dtype=float).reshape(1, -1)
    scaled = classifier.scaler.transform(x)
    masker = shap.maskers.Independent(np.zeros((1, scaled.shape[1])))
    explainer = shap.LinearExplainer(classifier.model, masker)
    real_shap = dict(zip(classifier.feature_columns, explainer(scaled).values[0]))

    for metric_name, contribution in ours.items():
        assert abs(contribution - real_shap[metric_name]) < 1e-9
