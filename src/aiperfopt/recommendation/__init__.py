"""Recommendation Engine (Section 7.1, item 9): a deliberately rule-based
(ADL-15), thin mapping from Diagnosis's standardized taxonomy category
(FR-9.1) to curated, traceable action suggestions (FR-9.3), never taking
automated action itself (FR-9.2).

- ``catalog``  -> RecommendationCatalog + load_recommendation_catalog() —
                  loads and validates aiperfopt/resources/recommendations.yaml
- ``service``  -> generate_recommendations() — the Phase 7 entrypoint
"""

from aiperfopt.recommendation.catalog import (
    RecommendationCatalog,
    RecommendedAction,
    load_recommendation_catalog,
)
from aiperfopt.recommendation.service import generate_recommendations

__all__ = [
    "RecommendationCatalog",
    "RecommendedAction",
    "load_recommendation_catalog",
    "generate_recommendations",
]
