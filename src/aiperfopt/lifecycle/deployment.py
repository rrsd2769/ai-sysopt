"""
The only place a model_version is ever activated or rolled back. Every
path through this module writes a :class:`ModelDeploymentRecord` to the
``model_deployments`` table, whether or not the outcome was "deployed" --
a rejected candidate is exactly as important to the audit trail as an accepted one.

"Never silently activate" means there is deliberately no
function anywhere in this codebase that activates a model_version
without a caller passing an explicit ``confirmed=True``. The API
layer is expected to gate that argument behind an actual user action
(a "confirm deployment" button/endpoint), not a config flag or default.
"""

from __future__ import annotations

import time
from pathlib import Path

from sqlalchemy import Engine

from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import ModelDeploymentError
from aiperfopt.config.settings import ForecastingSettings, RetrainingSettings
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.forecasting.baselines import Forecaster
from aiperfopt.forecasting.persistence import load_forecaster
from aiperfopt.forecasting.training_pipeline import MetricHorizonOutcome, baseline_factories_for
from aiperfopt.lifecycle.active_version import read_active_version, write_active_version
from aiperfopt.lifecycle.comparison import RetrainingComparison
from aiperfopt.lifecycle.gate import SkillGateResult
from aiperfopt.storage.reader import get_evaluations_by_version, get_model_deployments_range
from aiperfopt.storage.records import ModelDeploymentRecord
from aiperfopt.storage.writer import insert_model_deployment

logger = get_logger(__name__)


def get_active_model_version(retraining_settings: RetrainingSettings) -> str | None:
    """The model_version the live pipeline should currently be using, if any."""
    return read_active_version(retraining_settings.active_version_file)


def confirm_deployment(
    engine: Engine,
    retraining_settings: RetrainingSettings,
    candidate_version: str,
    comparison: RetrainingComparison,
    skill_gate: SkillGateResult,
    confirmed: bool,
    confirmed_by: str = "human",
) -> ModelDeploymentRecord:
    """Record a human's deployment decision for ``candidate_version``.

    Parameters
    ----------
    skill_gate:
        The absolute quality bar (:func:`aiperfopt.lifecycle.
        comparison.evaluate_skill_gate`), computed once alongside
        ``comparison`` when the candidate was produced. Unlike
        ``comparison.recommendation``, this is not advisory: a failing
        gate raises :class:`~aiperfopt.common.exceptions.
        ModelDeploymentError` below even when ``confirmed=True`` --
        human confirmation remains necessary but stops being sufficient
        (the precision-favoring, trust-first stance). The audit
        trail still records the attempt before raising, a
        "gate_failed" decision alongside "deployed"/"rejected"/"rollback".
    confirmed:
        The actual decision. This function does not infer a decision
        from ``comparison.recommendation`` -- the recommendation is
        advisory only; a human may confirm a "reject"
        recommendation or decline a "deploy" one, and either is recorded
        faithfully either way. Irrelevant to whether the skill gate
        blocks deployment: that check happens regardless of ``confirmed``.
    confirmed_by:
        Free-text identifying who/what confirmed this. The API
        layer should pass a real identifier once authentication exists
        beyond this single-user, local-only V1; "human" is
        an acceptable placeholder until then.

    Returns
    -------
    The persisted :class:`ModelDeploymentRecord`, with the active-version
    pointer file already updated if ``confirmed`` is True and the gate
    passed.

    Raises
    ------
    ModelDeploymentError
        If ``skill_gate.passed`` is False -- the active-version pointer
        is never written in this case, regardless of ``confirmed``.
    """
    previous_version = get_active_model_version(retraining_settings)

    if not skill_gate.passed:
        record = ModelDeploymentRecord(
            decided_at=time.time(),
            candidate_version=candidate_version,
            previous_version=previous_version,
            decision="gate_failed",
            recommendation=comparison.recommendation,
            confirmed_by=confirmed_by,
            comparison_summary={"comparison": comparison.as_dict(), "skill_gate": skill_gate.as_dict()},
        )
        record.id = insert_model_deployment(engine, record)
        logger.warning(
            "Deployment refused -- quality gate failed",
            extra={"candidate_version": candidate_version, "reasons": skill_gate.reasons},
        )
        raise ModelDeploymentError(
            f"Quality gate failed for candidate {candidate_version!r}: " + "; ".join(skill_gate.reasons),
            recoverable=True,
            context={"candidate_version": candidate_version, "reasons": skill_gate.reasons},
        )

    decision = "deployed" if confirmed else "rejected"

    if confirmed:
        write_active_version(retraining_settings.active_version_file, candidate_version)

    record = ModelDeploymentRecord(
        decided_at=time.time(),
        candidate_version=candidate_version,
        previous_version=previous_version,
        decision=decision,
        recommendation=comparison.recommendation,
        confirmed_by=confirmed_by,
        comparison_summary={"comparison": comparison.as_dict(), "skill_gate": skill_gate.as_dict()},
    )
    record.id = insert_model_deployment(engine, record)

    logger.info(
        "Deployment decision recorded",
        extra={
            "candidate_version": candidate_version, "previous_version": previous_version,
            "decision": decision, "recommendation": comparison.recommendation, "confirmed_by": confirmed_by,
        },
    )
    return record


def rollback_to_last_known_good(
    engine: Engine, retraining_settings: RetrainingSettings, reason: str
) -> ModelDeploymentRecord:
    """Automatic fallback to the last known-good deployed version.

    Walks the deployment history backwards from the current active
    version for the most recent *other* version whose decision was
    "deployed", and activates it. Raises :class:`ModelDeploymentError`
    (non-recoverable) if no earlier deployed version exists -- there is
    nothing to fall back to, and that is a condition the caller (the inference path) must surface rather than silently ignore.
    """
    current_version = get_active_model_version(retraining_settings)
    history = get_model_deployments_range(engine)  # chronological ascending

    prior_deployed = [
        d for d in reversed(history) if d.decision == "deployed" and d.candidate_version != current_version
    ]
    if not prior_deployed:
        raise ModelDeploymentError(
            "No previous known-good deployed model version available to roll back to",
            recoverable=False,
            context={"current_version": current_version, "reason": reason},
        )

    target_version = prior_deployed[0].candidate_version
    write_active_version(retraining_settings.active_version_file, target_version)

    record = ModelDeploymentRecord(
        decided_at=time.time(),
        candidate_version=target_version,
        previous_version=current_version,
        decision="rollback",
        recommendation=None,
        confirmed_by="system",
        comparison_summary={"reason": reason},
    )
    record.id = insert_model_deployment(engine, record)

    logger.warning(
        "Automatic rollback to last known-good model version",
        extra={"from_version": current_version, "to_version": target_version, "reason": reason},
    )
    return record


def load_deployed_forecaster(
    outcome: MetricHorizonOutcome, models_dir: Path, model_version: str
) -> Forecaster:
    """Return a ready-to-use forecaster matching what a training run selected."""
    if outcome.selected_forecaster == "lightgbm":
        return load_forecaster(models_dir, outcome.metric_name, outcome.horizon_label, model_version)

    factories = baseline_factories_for(outcome.metric_name, outcome.horizon_seconds, is_counter=outcome.is_counter)
    if outcome.selected_forecaster not in factories:
        raise ValueError(f"Unknown selected forecaster: {outcome.selected_forecaster!r}")
    return factories[outcome.selected_forecaster]()


def resolve_deployed_forecasters(
    engine: Engine,
    feature_config: FeatureEngineeringConfig,
    forecasting_settings: ForecastingSettings,
    model_version: str,
    horizon_label: str = "short",
    target_metrics: list[str] | None = None,
) -> dict[str, Forecaster]:
    """Reconstruct every deployed forecaster for ``model_version`` from persisted evaluation history.

    Skips (with no error) any metric that has no recorded holdout
    evaluation for ``model_version``/``horizon_label`` — e.g. a metric
    that produced no usable model during that training run (rare, but possible for a very short/sparse training window). A caller with
    zero usable forecasters back should treat that the same as "no
    active model" (graceful degradation), not raise.
    """
    metrics = target_metrics or list(feature_config.gauge_metrics) + list(feature_config.counter_metrics)
    evaluations = get_evaluations_by_version(engine, model_version)

    forecasters: dict[str, Forecaster] = {}
    horizon_seconds = (
        forecasting_settings.short_horizon_target_seconds
        if horizon_label == "short"
        else forecasting_settings.medium_horizon_target_seconds
    )

    for metric_name in metrics:
        matches = [
            e
            for e in evaluations
            if e.metric_name == metric_name
            and e.horizon_label == horizon_label
            and e.dataset_tag is not None
            and e.dataset_tag.startswith("holdout_")
        ]
        if not matches:
            continue

        latest = max(matches, key=lambda e: e.evaluated_at)
        selected_forecaster = latest.model_name
        is_counter = metric_name in feature_config.counter_metrics

        outcome = MetricHorizonOutcome(
            metric_name=metric_name, horizon_label=horizon_label, horizon_seconds=int(horizon_seconds),
            walk_forward_summary={}, selected_forecaster=selected_forecaster,
            holdout_mae=latest.mae, holdout_rmse=latest.rmse,
            holdout_directional_accuracy=latest.directional_accuracy, is_counter=is_counter,
        )
        forecasters[metric_name] = load_deployed_forecaster(
            outcome, forecasting_settings.models_dir, model_version
        )

    return forecasters
