"""
Absolute quality bar a candidate must clear before it can be deployed:
beating persistence (forecasting) and the threshold-rule baseline
(contention). Unlike :mod:`aiperfopt.lifecycle.comparison`, this blocks
deployment even when a human has confirmed it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aiperfopt.applog import get_logger
from aiperfopt.contention.training_pipeline import ContentionTrainingSummary
from aiperfopt.forecasting.training_pipeline import TrainingRunSummary

logger = get_logger(__name__)


# --- the quality gate ------------------------------------------------
#
# Deliberately a *different* comparison from everything above. The
# regression check above answers "is this candidate at least as good as
# whatever's currently deployed" -- comparative, and silent (recommends,
# never blocks) when there's no active version to compare against, which
# is exactly right for a first-ever deployment. The skill gate instead
# answers an absolute question that doesn't care what (if anything) is
# currently deployed: "does this candidate clear the minimum quality bar
# any deployed model must clear at all" -- beating the threshold-rule
# baseline (contention) or persistence (forecasting), the same baselines
# introduced to make an honest evaluation possible in the
# first place. Unlike the regression check, this one really does block:
# see :mod:`aiperfopt.lifecycle.deployment`, where it overrides even an
# explicit human ``confirmed=True`` (the precision-favoring,
# trust-first stance -- "nothing can fail" was the defect this
# gate exists to fix).


@dataclass
class ForecastGateCriterion:
    metric_name: str
    horizon_label: str
    candidate_mae: float | None
    persistence_mae: float | None
    candidate_directional_accuracy: float | None
    passed: bool
    note: str | None = None


@dataclass
class ContentionGateCriterion:
    candidate_f1: float | None
    baseline_f1: float | None
    passed: bool
    note: str | None = None


@dataclass
class SkillGateResult:
    passed: bool
    forecast_criteria: list[ForecastGateCriterion] = field(default_factory=list)
    contention_criterion: ContentionGateCriterion | None = None
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": self.reasons,
            "forecast_criteria": [
                {
                    "metric_name": c.metric_name, "horizon_label": c.horizon_label,
                    "candidate_mae": c.candidate_mae, "persistence_mae": c.persistence_mae,
                    "candidate_directional_accuracy": c.candidate_directional_accuracy,
                    "passed": c.passed, "note": c.note,
                }
                for c in self.forecast_criteria
            ],
            "contention_criterion": (
                None
                if self.contention_criterion is None
                else {
                    "candidate_f1": self.contention_criterion.candidate_f1,
                    "baseline_f1": self.contention_criterion.baseline_f1,
                    "passed": self.contention_criterion.passed,
                    "note": self.contention_criterion.note,
                }
            ),
        }


def _mae_improvement_over_persistence(candidate_mae: float, persistence_mae: float) -> float:
    """Relative MAE improvement, same convention as
    :func:`aiperfopt.forecasting.evaluation.select_best_forecaster_name`'s
    contingency-plan comparison -- reused here so "beats the baseline by
    a margin" means the same thing in both places. A non-positive
    persistence MAE (a metric that never moved in the holdout window) has
    no relative improvement to speak of; only exactly matching it counts.
    """
    if persistence_mae <= 0:
        return 0.0 if candidate_mae <= persistence_mae else -1.0
    return (persistence_mae - candidate_mae) / persistence_mae


def evaluate_skill_gate(
    forecast_summary: TrainingRunSummary,
    contention_summary: ContentionTrainingSummary | None,
    forecasting_mae_margin: float,
    directional_accuracy_margin: float,
    contention_f1_margin: float,
) -> SkillGateResult:
    """The absolute quality bar every deployable candidate must clear.

    Evaluated per target on the chronological holdout (never the
    walk-forward CV folds, which are a different data slice): forecasting
    outcomes need ``holdout_mae``/``holdout_persistence_mae`` and
    contention needs ``holdout_metrics``/``threshold_rule_baseline``, all
    already computed by the training pipelines. The gate is on
    the *deployed* model, whichever it is -- if a classical baseline
    genuinely won the walk-forward contingency-plan comparison, it faces
    the exact same "beats persistence" bar here as LightGBM would.

    A target with no holdout evaluation available (too little data, or a
    single-class holdout that never produced a trained classifier -- the
    honest single-class real-data result) is *not* a failure:
    there's nothing to measure, so it's silently excluded rather than
    forced to a pass or fail it can't earn. If that leaves nothing at all
    evaluable, the gate passes (nothing failed) but says so explicitly --
    it does not become an argument for blocking a deployment that
    the separate held-out-generalization check and human
    confirmation already govern.

    Only ``horizon_label == "short"`` forecast outcomes are gated. This
    mirrors :mod:`aiperfopt.contention.training_pipeline`'s own
    pre-existing scope decision (only short-horizon forecasts feed
    Contention Prediction, the headline deliverable; medium-horizon forecasting is trained and
    reported but retained as secondary) -- the quality gate blocks
    deployment of the bundle actually driving the product's headline
    behavior, not every auxiliary forecast trained alongside it.
    """
    result = SkillGateResult(passed=True)

    for outcome in forecast_summary.outcomes:
        if outcome.horizon_label != "short":
            continue
        if outcome.holdout_mae is None or outcome.holdout_persistence_mae is None:
            continue

        improvement = _mae_improvement_over_persistence(outcome.holdout_mae, outcome.holdout_persistence_mae)
        passes_mae = improvement >= forecasting_mae_margin
        passes_directional = (
            outcome.holdout_directional_accuracy is not None
            and outcome.holdout_directional_accuracy > 0.5 + directional_accuracy_margin
        )
        passed = passes_mae and passes_directional

        note = None
        if not passed:
            failed_parts = []
            if not passes_mae:
                failed_parts.append(f"MAE improvement over persistence {improvement:.1%} < required {forecasting_mae_margin:.1%}")
            if not passes_directional:
                directional_str = (
                    "n/a" if outcome.holdout_directional_accuracy is None
                    else f"{outcome.holdout_directional_accuracy:.2f}"
                )
                failed_parts.append(
                    f"directional accuracy {directional_str} <= required {0.5 + directional_accuracy_margin:.2f}"
                )
            note = "; ".join(failed_parts)
            result.passed = False
            result.reasons.append(f"{outcome.metric_name}/{outcome.horizon_label}: {note}")

        result.forecast_criteria.append(
            ForecastGateCriterion(
                metric_name=outcome.metric_name, horizon_label=outcome.horizon_label,
                candidate_mae=outcome.holdout_mae, persistence_mae=outcome.holdout_persistence_mae,
                candidate_directional_accuracy=outcome.holdout_directional_accuracy,
                passed=passed, note=note,
            )
        )

    if (
        contention_summary is not None
        and contention_summary.threshold_rule_baseline is not None
        and contention_summary.holdout_metrics
    ):
        candidate_f1 = contention_summary.holdout_metrics.get("f1")
        baseline_f1 = contention_summary.threshold_rule_baseline.get("f1")
        if candidate_f1 is not None and baseline_f1 is not None:
            # F1 is bounded above at 1.0, and the threshold-rule baseline
            # can legitimately be very good -- even perfect -- on an
            # easy/small/separable holdout (a monotonic trend, a clean
            # flat series): there, "beat an already-excellent
            # baseline by a margin" is not a quality signal, it's a
            # mathematical impossibility. The gate's actual job (Overview:
            # "gives the numbers integrity -- currently nothing
            # can fail") is catching a genuinely bad classifier, not
            # requiring the trained model to strictly dominate a simple
            # deterministic rule it may legitimately only tie. So this is
            # a tolerance on how far the candidate may trail the
            # baseline, not a required margin of superiority -- matching
            # the same "not worse than, within tolerance" idiom
            # ``_regressed_higher_is_worse`` already uses above for the
            # separate active-version regression check.
            required_f1 = baseline_f1 - contention_f1_margin
            passed = candidate_f1 >= required_f1
            note = None
            if not passed:
                note = (
                    f"F1 {candidate_f1:.3f} falls short of threshold-rule baseline {baseline_f1:.3f} "
                    f"by more than the allowed {contention_f1_margin:.3f}"
                )
                result.passed = False
                result.reasons.append(f"contention_classifier: {note}")
            result.contention_criterion = ContentionGateCriterion(
                candidate_f1=candidate_f1, baseline_f1=baseline_f1, passed=passed, note=note,
            )

    if not result.forecast_criteria and result.contention_criterion is None:
        result.reasons.append(
            "No holdout evaluation was available for any target -- the quality gate could not "
            "verify anything and is not blocking on that basis alone."
        )

    logger.info(
        "Skill gate evaluated",
        extra={"passed": result.passed, "reasons": result.reasons},
    )
    return result
