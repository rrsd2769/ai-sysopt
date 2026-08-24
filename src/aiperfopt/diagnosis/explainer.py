"""
Tiered explainability (Section 10.5, FR-8.1, NFR-1.3, NFR-9.1, ADL-14, ADL-21).

Section 10.5 requires explanation generation to be *tiered*: a lightweight
explanation available during routine monitoring, escalating to expensive
SHAP-class computation only on high-confidence contention predictions,
significant anomalies, or explicit Developer Mode request. This module is
where that tiering is actually implemented — but it applies meaningfully
to only *one* of the two attribution surfaces Diagnosis reconciles:

Forecasting attribution: genuinely tiered
-------------------------------------------
For a LightGBM forecaster (:class:`aiperfopt.forecasting.model.MetricForecaster`),
two real options exist with a real cost difference:

- **Lightweight**: the booster's own native ``feature_importance()`` — a
  global, training-time-computed ranking, effectively free to read at
  request time (NFR-1.3's "must not run on the default monitoring
  cycle" is trivially satisfied since nothing is computed per-request).
- **SHAP**: ``shap.TreeExplainer(booster).shap_values(row)`` — a genuine
  per-instance, per-request computation (ADL-14), more expensive and
  exactly what Section 10.5 means by "SHAP-class methods."

Section 10.7's contingency plan means the *deployed* forecaster for a
given metric/horizon may not be LightGBM at all (a baseline can win).
Baselines have no learned per-feature attribution to give at either
tier — :func:`explain_forecast` handles this by returning a
``tier="not_applicable"`` result with a plain textual note, rather than
either fabricating an attribution a formula-based baseline doesn't have,
or silently omitting that metric's explanation from Developer Mode.

Contention attribution: not meaningfully tiered
----------------------------------------------------
:mod:`aiperfopt.contention.model`'s own docstring already explains why
logistic regression was chosen over a small GBM for Contention Prediction:
its coefficients give free, exact per-feature attribution. This is not
merely *similar* to what SHAP would report for a linear model — it is
*identical*. For a linear model with a zero-centered (standardized)
background, SHAP's Kernel/Linear explainer decomposes a prediction as
``sum(coefficient_i * standardized_value_i)`` per feature, which is
exactly :meth:`aiperfopt.contention.model.ContentionClassifier.contributing_metrics`'s
existing formula. This equivalence is not just asserted here — it is
checked numerically in ``tests/test_diagnosis_explainer.py`` against a
real ``shap.LinearExplainer`` call, so a future refactor that breaks it
would fail a test, not just a stale docstring claim (see
:mod:`aiperfopt.storage.schema`'s ``contention_predictions`` column
comment for a cautionary example of exactly that kind of drift).
Consequently, Diagnosis never recomputes contention attribution here —
:mod:`aiperfopt.diagnosis.service` reuses
``ContentionPredictionRecord.contributing_metrics`` directly, tier or no
tier, and this module has no ``explain_contention`` function to avoid
implying a second, redundant code path exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from aiperfopt.forecasting.baselines import Forecaster
from aiperfopt.forecasting.model import MetricForecaster

EXPLANATION_TIER_LIGHTWEIGHT = "lightweight"
EXPLANATION_TIER_SHAP = "shap"
EXPLANATION_TIER_NOT_APPLICABLE = "not_applicable"

TRIGGER_HIGH_CONFIDENCE_CONTENTION = "high_confidence_contention"
TRIGGER_SIGNIFICANT_ANOMALY = "significant_anomaly"
TRIGGER_EXPLICIT_REQUEST = "explicit_request"
TRIGGER_ROUTINE = "routine"


@dataclass(frozen=True)
class FeatureAttribution:
    feature_name: str
    value: float  # signed: SHAP value or (for lightweight) native importance


@dataclass(frozen=True)
class ForecastExplanation:
    metric_name: str
    forecaster_kind: str  # "lightgbm" | the baseline class name
    tier: str  # EXPLANATION_TIER_*
    top_features: list[FeatureAttribution] = field(default_factory=list)
    note: str | None = None

    def to_payload(self) -> dict:
        """JSON-serializable form for the ``diagnoses.forecast_explanations`` column."""
        return {
            "metric_name": self.metric_name,
            "forecaster_kind": self.forecaster_kind,
            "tier": self.tier,
            "top_features": [{"feature_name": f.feature_name, "value": f.value} for f in self.top_features],
            "note": self.note,
        }


def should_use_shap_tier(
    triggered: bool,
    anomaly_score: float,
    anomaly_significant_threshold: float,
    explicit_request: bool,
) -> tuple[bool, str]:
    """Section 10.5's three SHAP triggers, in priority order.

    Returns ``(use_shap, trigger_reason)``. ``triggered`` (FR-6.3's
    conservative decision threshold already having cleared) stands in for
    "high-confidence contention prediction" — Section 11 already tunes
    that threshold conservatively in favor of precision, so a triggered
    prediction *is* the system's own definition of high-confidence; a
    separate, stricter "high confidence" threshold here would be a second
    number doing the same job the decision threshold already does.
    """
    if explicit_request:
        return True, TRIGGER_EXPLICIT_REQUEST
    if anomaly_score >= anomaly_significant_threshold:
        return True, TRIGGER_SIGNIFICANT_ANOMALY
    if triggered:
        return True, TRIGGER_HIGH_CONFIDENCE_CONTENTION
    return False, TRIGGER_ROUTINE


def _lightweight_forecast_explanation(forecaster: MetricForecaster, top_n: int) -> list[FeatureAttribution]:
    importances = forecaster.booster.feature_importance(importance_type="gain")
    paired = list(zip(forecaster.feature_columns, importances))
    paired.sort(key=lambda item: abs(item[1]), reverse=True)
    return [FeatureAttribution(name, float(value)) for name, value in paired[:top_n]]


def _shap_forecast_explanation(
    forecaster: MetricForecaster, feature_row: pd.Series, top_n: int
) -> list[FeatureAttribution]:
    import shap  # imported lazily: only the SHAP tier pays this import's cost

    x = feature_row.reindex(forecaster.feature_columns).to_frame().T.astype(float)
    explainer = shap.TreeExplainer(forecaster.booster)
    shap_values = explainer.shap_values(x)
    # shap_values is (1, n_features) for a single-output regressor.
    values = shap_values[0] if hasattr(shap_values, "__len__") and len(shap_values) == 1 else shap_values
    paired = list(zip(forecaster.feature_columns, list(values)))
    paired.sort(key=lambda item: abs(item[1]), reverse=True)
    return [FeatureAttribution(name, float(value)) for name, value in paired[:top_n]]


def explain_forecast(
    metric_name: str,
    forecaster: Forecaster,
    feature_row: pd.Series,
    use_shap: bool,
    top_n: int,
) -> ForecastExplanation:
    """Explain one metric's deployed forecaster at the appropriate tier.

    ``forecaster`` may be a :class:`MetricForecaster` (LightGBM) or any of
    :mod:`aiperfopt.forecasting.baselines`' baseline classes — only the
    former supports either explanation tier; see the module docstring.
    """
    if not isinstance(forecaster, MetricForecaster) or forecaster.booster is None:
        kind = type(forecaster).__name__
        return ForecastExplanation(
            metric_name=metric_name,
            forecaster_kind=kind,
            tier=EXPLANATION_TIER_NOT_APPLICABLE,
            note=(
                f"'{metric_name}' is currently forecast by the {kind} baseline, not the "
                "LightGBM model — Section 10.7's contingency plan selected it because it "
                "validated at least as well as LightGBM for this metric. It is a simple, "
                "formula-based extrapolation (not a learned model), so no per-feature "
                "attribution applies; the forecast itself is still shown, just without "
                "SHAP/feature-importance backing it."
            ),
        )

    if use_shap:
        return ForecastExplanation(
            metric_name=metric_name,
            forecaster_kind="lightgbm",
            tier=EXPLANATION_TIER_SHAP,
            top_features=_shap_forecast_explanation(forecaster, feature_row, top_n),
        )

    return ForecastExplanation(
        metric_name=metric_name,
        forecaster_kind="lightgbm",
        tier=EXPLANATION_TIER_LIGHTWEIGHT,
        top_features=_lightweight_forecast_explanation(forecaster, top_n),
    )
