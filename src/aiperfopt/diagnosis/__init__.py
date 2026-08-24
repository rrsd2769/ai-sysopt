"""Diagnosis / Explainable AI (Section 7.1, item 8): reconciles Forecasting
and Contention Prediction output into exactly one of Section 12's eight
taxonomy categories (FR-8.3, ADL-24), with tiered SHAP/native-importance
attribution (Section 10.5, FR-8.1) and a plain-language translation for
General Mode (FR-8.2).

- ``taxonomy``   -> DiagnosisCategory + resolve_diagnosis_category() — the taxonomy resolver
- ``explainer``  -> tiered lightweight/SHAP attribution for LightGBM forecasters
- ``text``       -> technical (Developer Mode) and plain-language (General Mode) explanations
- ``service``    -> diagnose_contention_event() — the Phase 6 entrypoint
"""

from aiperfopt.diagnosis.explainer import (
    EXPLANATION_TIER_LIGHTWEIGHT,
    EXPLANATION_TIER_NOT_APPLICABLE,
    EXPLANATION_TIER_SHAP,
    TRIGGER_EXPLICIT_REQUEST,
    TRIGGER_HIGH_CONFIDENCE_CONTENTION,
    TRIGGER_ROUTINE,
    TRIGGER_SIGNIFICANT_ANOMALY,
    FeatureAttribution,
    ForecastExplanation,
    explain_forecast,
    should_use_shap_tier,
)
from aiperfopt.diagnosis.service import diagnose_contention_event
from aiperfopt.diagnosis.taxonomy import (
    NON_ACTIONABLE_CATEGORIES,
    DiagnosisCategory,
    is_actionable,
    resolve_diagnosis_category,
)
from aiperfopt.diagnosis.text import build_plain_language_explanation, build_technical_explanation

__all__ = [
    "DiagnosisCategory",
    "NON_ACTIONABLE_CATEGORIES",
    "is_actionable",
    "resolve_diagnosis_category",
    "EXPLANATION_TIER_LIGHTWEIGHT",
    "EXPLANATION_TIER_NOT_APPLICABLE",
    "EXPLANATION_TIER_SHAP",
    "TRIGGER_EXPLICIT_REQUEST",
    "TRIGGER_HIGH_CONFIDENCE_CONTENTION",
    "TRIGGER_ROUTINE",
    "TRIGGER_SIGNIFICANT_ANOMALY",
    "FeatureAttribution",
    "ForecastExplanation",
    "explain_forecast",
    "should_use_shap_tier",
    "build_plain_language_explanation",
    "build_technical_explanation",
    "diagnose_contention_event",
]
