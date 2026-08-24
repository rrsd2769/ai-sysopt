"""
Recommendation Engine orchestrator — the Phase 7 entrypoint (Section 7.1
item 9, FR-9.1-9.3, ADL-15, ADL-19).

:func:`generate_recommendations` is intentionally the thinnest orchestrator
in this codebase: gate on :attr:`~aiperfopt.storage.records.DiagnosisRecord.is_actionable`,
look the category up in the :class:`~aiperfopt.recommendation.catalog.RecommendationCatalog`,
persist, return. There is no branching logic here beyond that gate — all
of the actual decision-making (which actions map to which category) lives
in the externally-editable catalog (NFR-12.1), and all of the upstream
reconciliation work (which category this event even *is*) already
happened in Phase 6. This is deliberate: FR-9.1 requires the
Recommendation Engine consume *only* the standardized diagnosis category,
never raw metrics or ad hoc conditions, and Section 7.3 draws the
dependency as one-directional (Recommendation depends on Diagnosis'
output, never reaches past it to Contention Prediction or Forecasting
directly) — a thin orchestrator is what actually enforces that boundary
in code, rather than just documenting it.

FR-9.2 ("never take automated action... suggestions only, always
user-confirmed") is satisfied by what this module does *not* do: nothing
here touches a running process, changes a setting, or performs any
system-level action. It produces and persists data — a ranked list of
suggested actions — and nothing more. Any future automation would be a
different, explicitly-scoped subsystem, not an extension of this one.
"""

from __future__ import annotations

from sqlalchemy import Engine

from aiperfopt.recommendation.catalog import RecommendationCatalog
from aiperfopt.storage.records import DiagnosisRecord, RecommendationRecord
from aiperfopt.storage.writer import insert_recommendation


def generate_recommendations(
    engine: Engine,
    diagnosis: DiagnosisRecord,
    catalog: RecommendationCatalog,
) -> RecommendationRecord | None:
    """Look up, persist, and return curated actions for an actionable diagnosis.

    Returns ``None`` (persisting nothing) when ``diagnosis.is_actionable``
    is ``False`` — currently only true for the anomaly category (ADL-36).
    Checking the boolean field directly, rather than re-deriving
    actionability from ``diagnosis.category`` here, keeps this module
    agnostic to which categories are actionable: that decision was already
    made once, in :mod:`aiperfopt.diagnosis.taxonomy`, and shouldn't be
    made a second, potentially-inconsistent way here.
    """
    if not diagnosis.is_actionable:
        return None
    if diagnosis.id is None:
        raise ValueError(
            "diagnosis.id is required to generate traceable recommendations (FR-9.3) — "
            "pass a DiagnosisRecord that has already been persisted via insert_diagnosis()."
        )

    actions = catalog.actions_for(diagnosis.category)
    record = RecommendationRecord(
        generated_at=diagnosis.generated_at,
        diagnosis_id=diagnosis.id,
        diagnosis_category=diagnosis.category,
        actions=[action.to_payload() for action in actions],
        catalog_version=catalog.catalog_version,
    )
    insert_recommendation(engine, record)
    return record
