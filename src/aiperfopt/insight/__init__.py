"""Diagnosis and recommendations: reconciles forecasting and contention output
into one of the eight taxonomy categories, explains it (tiered SHAP or native
importance, technical and plain-language text), and maps actionable categories
to curated, rule-based action suggestions that are never applied automatically.

- ``taxonomy``        -> DiagnosisCategory + resolve_diagnosis_category()
- ``explainer``       -> tiered lightweight/SHAP attribution for LightGBM forecasters
- ``text``            -> technical (Developer Mode) and plain-language (General Mode) explanations
- ``diagnosis``       -> diagnose_contention_event()
- ``catalog``         -> RecommendationCatalog, loads resources/recommendations.yaml
- ``recommendations`` -> generate_recommendations()
"""

from aiperfopt.insight.explainer import (
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
from aiperfopt.insight.catalog import (
    RecommendationCatalog,
    RecommendedAction,
    load_recommendation_catalog,
)
from aiperfopt.insight.diagnosis import diagnose_contention_event
from aiperfopt.insight.recommendations import generate_recommendations
from aiperfopt.insight.taxonomy import (
    NON_ACTIONABLE_CATEGORIES,
    DiagnosisCategory,
    is_actionable,
    resolve_diagnosis_category,
)
from aiperfopt.insight.text import build_plain_language_explanation, build_technical_explanation

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
    "RecommendationCatalog",
    "RecommendedAction",
    "load_recommendation_catalog",
    "generate_recommendations",
]
