from __future__ import annotations

import pytest
from sqlalchemy import Engine

from aiperfopt.recommendation import generate_recommendations, load_recommendation_catalog
from aiperfopt.storage import DiagnosisRecord, get_recommendations_range, insert_diagnosis

_CATALOG = load_recommendation_catalog()


def _persisted_diagnosis(engine: Engine, **overrides) -> DiagnosisRecord:
    defaults = dict(
        generated_at=100.0,
        category="cpu_bound_single_process",
        is_actionable=True,
        confidence=0.9,
        explanation_tier="shap",
        trigger_reason="high_confidence_contention",
        technical_explanation="tech",
        plain_language_explanation="plain",
        model_version="v1",
    )
    defaults.update(overrides)
    record = DiagnosisRecord(**defaults)
    record.id = insert_diagnosis(engine, record)
    return record


def test_actionable_diagnosis_produces_and_persists_a_recommendation(storage_engine: Engine):
    diagnosis = _persisted_diagnosis(storage_engine)
    result = generate_recommendations(storage_engine, diagnosis, _CATALOG)

    assert result is not None
    assert result.diagnosis_id == diagnosis.id
    assert result.diagnosis_category == "cpu_bound_single_process"
    assert result.catalog_version == _CATALOG.catalog_version
    assert len(result.actions) >= 2

    persisted = get_recommendations_range(storage_engine)
    assert len(persisted) == 1
    assert persisted[0].diagnosis_id == diagnosis.id


def test_non_actionable_diagnosis_returns_none_and_persists_nothing(storage_engine: Engine):
    diagnosis = _persisted_diagnosis(
        storage_engine,
        category="unusual_pattern_anomaly_flagged",
        is_actionable=False,
        confidence=0.9,
    )
    result = generate_recommendations(storage_engine, diagnosis, _CATALOG)

    assert result is None
    assert get_recommendations_range(storage_engine) == []


def test_every_actionable_category_round_trips_through_the_real_catalog(storage_engine: Engine):
    """Not just the catalog's own validation (test_recommendation_catalog.py) —
    this exercises the actual service path for every category the way a
    real Diagnosis event would produce it."""
    for category in _CATALOG.actions_by_category:
        diagnosis = _persisted_diagnosis(storage_engine, category=category, generated_at=float(len(category)))
        result = generate_recommendations(storage_engine, diagnosis, _CATALOG)
        assert result is not None
        assert result.diagnosis_category == category


def test_unpersisted_diagnosis_without_id_raises(storage_engine: Engine):
    diagnosis = DiagnosisRecord(
        generated_at=100.0, category="cpu_bound_single_process", is_actionable=True,
        confidence=0.9, explanation_tier="shap", trigger_reason="high_confidence_contention",
        technical_explanation="tech", plain_language_explanation="plain", model_version="v1",
    )
    assert diagnosis.id is None
    with pytest.raises(ValueError):
        generate_recommendations(storage_engine, diagnosis, _CATALOG)
