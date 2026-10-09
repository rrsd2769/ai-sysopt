"""
Recommendation catalog loading.

The catalog is the *entire* decision surface of the Recommendation
Engine: a pure ``category -> ranked actions`` mapping loaded from
``aiperfopt/resources/recommendations.yaml``. Deliberately not a Python dict literal
in source — operational parameters are externally
configurable without touching application source, and recommendation
wording is exactly the kind of thing that legitimately needs editing
without a code change (unlike, say, the taxonomy categories themselves,
which are a fixed set this module treats as a closed
:class:`~aiperfopt.insight.taxonomy.DiagnosisCategory` enum, not a
free-form string key).

Validated eagerly at load time, not lazily at lookup time
--------------------------------------------------------------
:func:`load_recommendation_catalog` checks the file covers every
*actionable* category (all of the eight except
``unusual_pattern_anomaly_flagged``) and rejects an entry for
the anomaly category outright. A catalog that's missing a category or
mistakenly includes the anomaly one is a misconfiguration that should
fail loudly at startup, not surface as a confusing ``None``/exception the
first time that category happens to come up live — the same "fail fast
on a bad config rather than degrade silently" instinct
:mod:`aiperfopt.config.settings` already applies to layered configuration
generally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aiperfopt.resources import RECOMMENDATION_CATALOG_FILENAME, resource_path
from typing import Any

import yaml

from aiperfopt.common.exceptions import RecommendationError
from aiperfopt.insight.taxonomy import DiagnosisCategory

_DEFAULT_CATALOG_PATH = resource_path(RECOMMENDATION_CATALOG_FILENAME)

# Every category the catalog must cover — every category except the
# anomaly one, which must never appear (see module docstring).
_REQUIRED_CATEGORIES = frozenset(
    category.value for category in DiagnosisCategory if category != DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY
)
_FORBIDDEN_CATEGORY = DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY.value


@dataclass(frozen=True)
class RecommendedAction:
    title: str
    description: str
    rationale: str

    def to_payload(self) -> dict[str, str]:
        return {"title": self.title, "description": self.description, "rationale": self.rationale}


@dataclass(frozen=True)
class RecommendationCatalog:
    catalog_version: str
    actions_by_category: dict[str, list[RecommendedAction]]

    def actions_for(self, category: str) -> list[RecommendedAction]:
        if category == _FORBIDDEN_CATEGORY:
            # Defense in depth: aiperfopt.insight.recommendations already
            # gates on DiagnosisRecord.is_actionable before ever calling
            # this, but a catalog lookup for the anomaly category should
            # still refuse outright rather than trust the caller checked.
            raise RecommendationError(
                f"'{_FORBIDDEN_CATEGORY}' is informational-only and must never "
                "receive a recommendation lookup.",
                recoverable=False,
                context={"category": category},
            )
        try:
            return self.actions_by_category[category]
        except KeyError as exc:
            raise RecommendationError(
                f"No recommendation catalog entry for diagnosis category '{category}'.",
                recoverable=False,
                context={"category": category},
            ) from exc


def _parse_action(category: str, raw: Any) -> RecommendedAction:
    if not isinstance(raw, dict) or not {"title", "description", "rationale"} <= raw.keys():
        raise RecommendationError(
            f"Malformed action entry for category '{category}': each action needs "
            "'title', 'description', and 'rationale'.",
            recoverable=False,
            context={"category": category, "entry": raw},
        )
    return RecommendedAction(title=str(raw["title"]), description=str(raw["description"]), rationale=str(raw["rationale"]))


def load_recommendation_catalog(path: Path | str | None = None) -> RecommendationCatalog:
    """Load and validate the recommendation catalog (defaults to the packaged file)."""
    catalog_path = Path(path) if path is not None else _DEFAULT_CATALOG_PATH
    try:
        with catalog_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RecommendationError(
            f"Failed to load recommendation catalog: {catalog_path}",
            recoverable=False,
            context={"path": str(catalog_path)},
        ) from exc

    if not isinstance(data, dict) or "catalog_version" not in data or "categories" not in data:
        raise RecommendationError(
            f"Recommendation catalog must contain 'catalog_version' and 'categories': {catalog_path}",
            recoverable=False,
            context={"path": str(catalog_path)},
        )

    categories = data["categories"]
    if not isinstance(categories, dict):
        raise RecommendationError(
            f"Recommendation catalog's 'categories' must be a mapping: {catalog_path}",
            recoverable=False,
            context={"path": str(catalog_path)},
        )

    if _FORBIDDEN_CATEGORY in categories:
        raise RecommendationError(
            f"Recommendation catalog must not contain an entry for '{_FORBIDDEN_CATEGORY}' "
            "(informational-only, never actionable).",
            recoverable=False,
            context={"path": str(catalog_path)},
        )

    missing = _REQUIRED_CATEGORIES - set(categories.keys())
    if missing:
        raise RecommendationError(
            f"Recommendation catalog is missing required categories: {sorted(missing)}",
            recoverable=False,
            context={"path": str(catalog_path), "missing": sorted(missing)},
        )

    unknown = set(categories.keys()) - _REQUIRED_CATEGORIES
    if unknown:
        raise RecommendationError(
            f"Recommendation catalog contains unrecognized categories not in the "
            f"taxonomy: {sorted(unknown)}",
            recoverable=False,
            context={"path": str(catalog_path), "unknown": sorted(unknown)},
        )

    actions_by_category: dict[str, list[RecommendedAction]] = {}
    for category, raw_actions in categories.items():
        if not isinstance(raw_actions, list) or not raw_actions:
            raise RecommendationError(
                f"Category '{category}' must map to a non-empty list of actions.",
                recoverable=False,
                context={"category": category},
            )
        actions_by_category[category] = [_parse_action(category, raw) for raw in raw_actions]

    return RecommendationCatalog(catalog_version=str(data["catalog_version"]), actions_by_category=actions_by_category)
