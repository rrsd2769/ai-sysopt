from __future__ import annotations

import pytest

from aiperfopt.common.exceptions import RecommendationError
from aiperfopt.insight.taxonomy import DiagnosisCategory
from aiperfopt.insight.catalog import load_recommendation_catalog


@pytest.mark.behavioural
def test_packaged_catalog_loads_and_covers_every_actionable_category():
    catalog = load_recommendation_catalog()
    assert catalog.catalog_version
    expected = {c.value for c in DiagnosisCategory if c != DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY}
    assert set(catalog.actions_by_category.keys()) == expected


@pytest.mark.behavioural
def test_packaged_catalog_has_no_anomaly_entry():
    catalog = load_recommendation_catalog()
    assert DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY.value not in catalog.actions_by_category


@pytest.mark.behavioural
def test_every_category_has_at_least_two_ranked_actions():
    catalog = load_recommendation_catalog()
    for category, actions in catalog.actions_by_category.items():
        assert len(actions) >= 2, category
        for action in actions:
            assert action.title and action.description and action.rationale


@pytest.mark.behavioural
def test_actions_for_returns_actions_in_file_order():
    catalog = load_recommendation_catalog()
    actions = catalog.actions_for(DiagnosisCategory.CPU_SINGLE_PROCESS.value)
    assert actions[0].title == "Quit or pause the dominant process"


@pytest.mark.plumbing
def test_actions_for_refuses_anomaly_category_even_if_asked():
    catalog = load_recommendation_catalog()
    with pytest.raises(RecommendationError):
        catalog.actions_for(DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY.value)


@pytest.mark.plumbing
def test_actions_for_unknown_category_raises():
    catalog = load_recommendation_catalog()
    with pytest.raises(RecommendationError):
        catalog.actions_for("not_a_real_category")


@pytest.mark.plumbing
def test_missing_file_raises_recommendation_error(tmp_path):
    with pytest.raises(RecommendationError):
        load_recommendation_catalog(tmp_path / "does_not_exist.yaml")


@pytest.mark.plumbing
def test_catalog_missing_a_required_category_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "catalog_version: 'v1'\n"
        "categories:\n"
        "  cpu_bound_single_process:\n"
        "    - title: t\n"
        "      description: d\n"
        "      rationale: r\n"
    )
    with pytest.raises(RecommendationError):
        load_recommendation_catalog(bad)


@pytest.mark.plumbing
def test_catalog_with_anomaly_entry_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    categories = "\n".join(
        f"  {c.value}:\n    - title: t\n      description: d\n      rationale: r"
        for c in DiagnosisCategory
    )
    bad.write_text(f"catalog_version: 'v1'\ncategories:\n{categories}\n")
    with pytest.raises(RecommendationError):
        load_recommendation_catalog(bad)


@pytest.mark.plumbing
def test_catalog_with_malformed_action_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    categories = "\n".join(
        f"  {c.value}:\n    - title: t\n      description: d\n      rationale: r"
        for c in DiagnosisCategory
        if c != DiagnosisCategory.UNUSUAL_PATTERN_ANOMALY
    )
    # Overwrite one category with a malformed entry (missing 'rationale').
    bad_categories = categories.replace(
        "cpu_bound_single_process:\n    - title: t\n      description: d\n      rationale: r",
        "cpu_bound_single_process:\n    - title: t\n      description: d\n",
    )
    bad.write_text(f"catalog_version: 'v1'\ncategories:\n{bad_categories}\n")
    with pytest.raises(RecommendationError):
        load_recommendation_catalog(bad)
