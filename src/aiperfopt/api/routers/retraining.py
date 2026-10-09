"""
Exposes Phase 8's Model Retraining subsystem over REST. This module adds
no retraining logic of its own — it validates a request, calls straight
into :mod:`aiperfopt.retraining`, and maps the result onto a response
schema. FR-11.2's "never silently activate" is enforced by
:func:`aiperfopt.retraining.deployment.confirm_deployment` itself
(``confirmed`` has no default there either); this router does not add or
relax that requirement, only exposes it.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from aiperfopt.api.dependencies import get_app_context
from aiperfopt.api.schemas import (
    ConfirmDeploymentRequest,
    ContentionComparisonOut,
    ContentionGateCriterionOut,
    ForecastGateCriterionOut,
    MetricComparisonOut,
    ModelDeploymentOut,
    RetrainingCandidateOut,
    RetrainingComparisonOut,
    RetrainRequest,
    RollbackRequest,
    SkillGateOut,
)
from aiperfopt.bootstrap import AppContext
from aiperfopt.common.exceptions import ModelDeploymentError, RetrainingError
from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.retraining.comparison import (
    ContentionComparison,
    ContentionGateCriterion,
    ForecastGateCriterion,
    MetricComparison,
    RetrainingComparison,
    SkillGateResult,
)
from aiperfopt.retraining.deployment import (
    confirm_deployment,
    get_active_model_version,
    rollback_to_last_known_good,
)
from aiperfopt.retraining.pipeline import run_retraining_on_real_usage_data

router = APIRouter(prefix="/retraining", tags=["retraining"])


def _comparison_out(comparison: RetrainingComparison) -> RetrainingComparisonOut:
    return RetrainingComparisonOut(
        candidate_version=comparison.candidate_version,
        active_version=comparison.active_version,
        recommendation=comparison.recommendation,
        reasons=comparison.reasons,
        metric_comparisons=[
            MetricComparisonOut(
                metric_name=c.metric_name, horizon_label=c.horizon_label,
                candidate_mae=c.candidate_mae, baseline_mae=c.baseline_mae,
                candidate_directional_accuracy=c.candidate_directional_accuracy,
                baseline_directional_accuracy=c.baseline_directional_accuracy,
                regressed=c.regressed, note=c.note,
            )
            for c in comparison.metric_comparisons
        ],
        contention_comparison=(
            None
            if comparison.contention_comparison is None
            else ContentionComparisonOut(
                candidate_f1=comparison.contention_comparison.candidate_f1,
                baseline_f1=comparison.contention_comparison.baseline_f1,
                regressed=comparison.contention_comparison.regressed,
                note=comparison.contention_comparison.note,
            )
        ),
    )


def _comparison_from_request(comparison: RetrainingComparisonOut) -> RetrainingComparison:
    """Reconstruct a domain :class:`RetrainingComparison` from what the client sends back.

    The confirm endpoint is deliberately stateless across requests -- the
    client round-trips the exact comparison it was shown (from
    ``/retraining/run``'s response) rather than this server re-deriving
    or re-fetching it, so what a human confirms is provably the same
    evidence they reviewed (ADL-23's human-confirmation requirement would
    be weaker if the server could silently substitute a fresher
    comparison between "run" and "confirm").
    """
    return RetrainingComparison(
        candidate_version=comparison.candidate_version,
        active_version=comparison.active_version,
        recommendation=comparison.recommendation,
        reasons=comparison.reasons,
        metric_comparisons=[
            MetricComparison(
                metric_name=c.metric_name, horizon_label=c.horizon_label,
                candidate_mae=c.candidate_mae, baseline_mae=c.baseline_mae,
                candidate_directional_accuracy=c.candidate_directional_accuracy,
                baseline_directional_accuracy=c.baseline_directional_accuracy,
                regressed=c.regressed, note=c.note,
            )
            for c in comparison.metric_comparisons
        ],
        contention_comparison=(
            None
            if comparison.contention_comparison is None
            else ContentionComparison(
                candidate_f1=comparison.contention_comparison.candidate_f1,
                baseline_f1=comparison.contention_comparison.baseline_f1,
                regressed=comparison.contention_comparison.regressed,
                note=comparison.contention_comparison.note,
            )
        ),
    )


def _gate_out(gate: SkillGateResult) -> SkillGateOut:
    return SkillGateOut(
        passed=gate.passed,
        reasons=gate.reasons,
        forecast_criteria=[
            ForecastGateCriterionOut(
                metric_name=c.metric_name, horizon_label=c.horizon_label,
                candidate_mae=c.candidate_mae, persistence_mae=c.persistence_mae,
                candidate_directional_accuracy=c.candidate_directional_accuracy,
                passed=c.passed, note=c.note,
            )
            for c in gate.forecast_criteria
        ],
        contention_criterion=(
            None
            if gate.contention_criterion is None
            else ContentionGateCriterionOut(
                candidate_f1=gate.contention_criterion.candidate_f1,
                baseline_f1=gate.contention_criterion.baseline_f1,
                passed=gate.contention_criterion.passed, note=gate.contention_criterion.note,
            )
        ),
    )


def _gate_from_request(gate: SkillGateOut) -> SkillGateResult:
    """Reconstruct a domain :class:`SkillGateResult` from what the client sends
    back -- same round-trip-what-was-shown rationale as
    :func:`_comparison_from_request` (ADL-23)."""
    return SkillGateResult(
        passed=gate.passed,
        reasons=gate.reasons,
        forecast_criteria=[
            ForecastGateCriterion(
                metric_name=c.metric_name, horizon_label=c.horizon_label,
                candidate_mae=c.candidate_mae, persistence_mae=c.persistence_mae,
                candidate_directional_accuracy=c.candidate_directional_accuracy,
                passed=c.passed, note=c.note,
            )
            for c in gate.forecast_criteria
        ],
        contention_criterion=(
            None
            if gate.contention_criterion is None
            else ContentionGateCriterion(
                candidate_f1=gate.contention_criterion.candidate_f1,
                baseline_f1=gate.contention_criterion.baseline_f1,
                passed=gate.contention_criterion.passed, note=gate.contention_criterion.note,
            )
        ),
    )


@router.get("/active-version")
def active_version(ctx: Annotated[AppContext, Depends(get_app_context)]) -> dict:
    return {"active_model_version": get_active_model_version(ctx.settings.retraining)}


@router.post("/run", response_model=RetrainingCandidateOut)
def run_retraining(body: RetrainRequest, ctx: Annotated[AppContext, Depends(get_app_context)]):
    feature_config = FeatureEngineeringConfig()
    labeling_config = ContentionLabelingConfig()
    try:
        candidate = run_retraining_on_real_usage_data(
            ctx.storage_engine, feature_config, ctx.settings.forecasting, ctx.settings.contention,
            labeling_config, ctx.settings.retraining, new_model_version=body.new_model_version,
            target_metrics=body.target_metrics, start=body.start, end=body.end,
        )
    except RetrainingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return RetrainingCandidateOut(
        model_version=candidate.model_version, data_source=candidate.data_source,
        comparison=_comparison_out(candidate.comparison), gate=_gate_out(candidate.skill_gate),
    )


@router.post("/confirm", response_model=ModelDeploymentOut)
def confirm(body: ConfirmDeploymentRequest, ctx: Annotated[AppContext, Depends(get_app_context)]):
    comparison = _comparison_from_request(body.comparison)
    gate = _gate_from_request(body.gate)
    try:
        record = confirm_deployment(
            ctx.storage_engine, ctx.settings.retraining, body.candidate_version, comparison, gate,
            confirmed=body.confirmed, confirmed_by=body.confirmed_by,
        )
    except ModelDeploymentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return record


@router.post("/rollback", response_model=ModelDeploymentOut)
def rollback(body: RollbackRequest, ctx: Annotated[AppContext, Depends(get_app_context)]):
    try:
        return rollback_to_last_known_good(ctx.storage_engine, ctx.settings.retraining, reason=body.reason)
    except ModelDeploymentError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
