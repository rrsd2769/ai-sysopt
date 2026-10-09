"""
Compares a freshly-trained retraining candidate against whatever model
version is currently deployed, and produces a recommendation — never a
decision. Deployment always requires a separate, explicit human
confirmation step (:mod:`aiperfopt.lifecycle.deployment`); this
module only answers "does the evidence support deploying this candidate".

What "regressed" means here
----------------------------
A candidate's holdout MAE/RMSE/F1/directional-accuracy is compared to the
active version's own previously-recorded holdout evaluation for the same
(metric, horizon) or contention classifier. A regression beyond
``regression_tolerance`` on *any* comparable outcome blocks the "deploy"
recommendation — this mirrors the precision-favoring,
trust-first stance: a candidate that's better on nine metrics and
meaningfully worse on one is not an unambiguous improvement, and a false
"upgrade" that quietly regresses one forecast is exactly the kind of
silent erosion of trust the project's evaluation philosophy exists to
prevent.

Where no baseline data exists for a given outcome (e.g. the active
version never trained that metric, or has no active version at all yet),
that outcome is reported but never counts against the recommendation —
there is nothing to regress relative to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Engine

from aiperfopt.applog import get_logger
from aiperfopt.contention.training_pipeline import ContentionTrainingSummary
from aiperfopt.forecasting.training_pipeline import TrainingRunSummary
from aiperfopt.storage.reader import get_evaluations_by_version

logger = get_logger(__name__)

Recommendation = str  # "deploy" | "reject" | "no_active_baseline"


@dataclass
class MetricComparison:
    metric_name: str
    horizon_label: str
    candidate_mae: float | None
    baseline_mae: float | None
    candidate_directional_accuracy: float | None
    baseline_directional_accuracy: float | None
    regressed: bool
    note: str | None = None


@dataclass
class ContentionComparison:
    candidate_f1: float | None
    baseline_f1: float | None
    regressed: bool
    note: str | None = None


@dataclass
class RetrainingComparison:
    candidate_version: str
    active_version: str | None
    metric_comparisons: list[MetricComparison] = field(default_factory=list)
    contention_comparison: ContentionComparison | None = None
    recommendation: Recommendation = "no_active_baseline"
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable form, for ``model_deployments.comparison_summary``."""
        return {
            "candidate_version": self.candidate_version,
            "active_version": self.active_version,
            "recommendation": self.recommendation,
            "reasons": self.reasons,
            "metric_comparisons": [
                {
                    "metric_name": c.metric_name,
                    "horizon_label": c.horizon_label,
                    "candidate_mae": c.candidate_mae,
                    "baseline_mae": c.baseline_mae,
                    "candidate_directional_accuracy": c.candidate_directional_accuracy,
                    "baseline_directional_accuracy": c.baseline_directional_accuracy,
                    "regressed": c.regressed,
                    "note": c.note,
                }
                for c in self.metric_comparisons
            ],
            "contention_comparison": (
                None
                if self.contention_comparison is None
                else {
                    "candidate_f1": self.contention_comparison.candidate_f1,
                    "baseline_f1": self.contention_comparison.baseline_f1,
                    "regressed": self.contention_comparison.regressed,
                    "note": self.contention_comparison.note,
                }
            ),
        }


def _regressed_lower_is_worse(candidate: float, baseline: float, tolerance: float) -> bool:
    """True if a metric where *lower* is better (MAE, RMSE) got worse beyond tolerance."""
    if baseline <= 0:
        return candidate > baseline
    return candidate > baseline * (1.0 + tolerance)


def _regressed_higher_is_worse(candidate: float, baseline: float, tolerance: float) -> bool:
    """True if a metric where *higher* is better (directional accuracy, F1) got worse beyond tolerance."""
    if baseline <= 0:
        return False
    return candidate < baseline * (1.0 - tolerance)


def _find_baseline_holdout_mae_and_directional(
    baseline_evaluations, metric_name: str, horizon_label: str
) -> tuple[float | None, float | None]:
    matches = [
        e
        for e in baseline_evaluations
        if e.metric_name == metric_name
        and e.horizon_label == horizon_label
        and e.dataset_tag is not None
        and e.dataset_tag.startswith("holdout_")
    ]
    if not matches:
        return None, None
    # Most recent matching holdout evaluation for that (metric, horizon).
    latest = max(matches, key=lambda e: e.evaluated_at)
    return latest.mae, latest.directional_accuracy


def _find_baseline_contention_f1(baseline_evaluations) -> float | None:
    matches = [
        e
        for e in baseline_evaluations
        if e.model_name == "contention_classifier"
        and e.dataset_tag is not None
        and e.dataset_tag.startswith("holdout_")
    ]
    if not matches:
        return None
    latest = max(matches, key=lambda e: e.evaluated_at)
    if latest.extra is None:
        return None
    return latest.extra.get("f1")


def compare_candidate_to_active(
    engine: Engine,
    candidate_version: str,
    forecast_summary: TrainingRunSummary,
    contention_summary: ContentionTrainingSummary | None,
    active_version: str | None,
    regression_tolerance: float,
) -> RetrainingComparison:
    """Build a :class:`RetrainingComparison` recommending deploy/reject.

    ``active_version`` is normally the result of
    :func:`aiperfopt.lifecycle.deployment.get_active_model_version` —
    passed in explicitly here (rather than read internally) to keep this
    module free of file I/O and independently testable.
    """
    comparison = RetrainingComparison(candidate_version=candidate_version, active_version=active_version)

    if active_version is None:
        comparison.recommendation = "no_active_baseline"
        comparison.reasons.append(
            "No model is currently active -- nothing to compare against. "
            "This would be the first deployment; still requires human confirmation."
        )
        # Still populate comparisons (with no baseline) so the full
        # candidate quality picture is visible to whoever confirms.
        baseline_evaluations: list = []
    else:
        baseline_evaluations = get_evaluations_by_version(engine, active_version)

    any_regression = False
    any_comparable = False

    for outcome in forecast_summary.outcomes:
        if outcome.holdout_mae is None:
            continue
        baseline_mae, baseline_directional = _find_baseline_holdout_mae_and_directional(
            baseline_evaluations, outcome.metric_name, outcome.horizon_label
        )
        if baseline_mae is None:
            comparison.metric_comparisons.append(
                MetricComparison(
                    metric_name=outcome.metric_name, horizon_label=outcome.horizon_label,
                    candidate_mae=outcome.holdout_mae, baseline_mae=None,
                    candidate_directional_accuracy=outcome.holdout_directional_accuracy,
                    baseline_directional_accuracy=None, regressed=False,
                    note="no baseline holdout evaluation available for this metric/horizon",
                )
            )
            continue

        any_comparable = True
        regressed = _regressed_lower_is_worse(outcome.holdout_mae, baseline_mae, regression_tolerance)
        if (
            not regressed
            and outcome.holdout_directional_accuracy is not None
            and baseline_directional is not None
        ):
            regressed = _regressed_higher_is_worse(
                outcome.holdout_directional_accuracy, baseline_directional, regression_tolerance
            )
        if regressed:
            any_regression = True
            comparison.reasons.append(
                f"{outcome.metric_name}/{outcome.horizon_label}: holdout MAE/directional accuracy "
                f"regressed beyond {regression_tolerance:.0%} tolerance versus active version {active_version}"
            )

        comparison.metric_comparisons.append(
            MetricComparison(
                metric_name=outcome.metric_name, horizon_label=outcome.horizon_label,
                candidate_mae=outcome.holdout_mae, baseline_mae=baseline_mae,
                candidate_directional_accuracy=outcome.holdout_directional_accuracy,
                baseline_directional_accuracy=baseline_directional, regressed=regressed,
            )
        )

    if contention_summary is not None:
        candidate_f1 = contention_summary.holdout_metrics.get("f1")
        if candidate_f1 is not None:
            baseline_f1 = _find_baseline_contention_f1(baseline_evaluations)
            if baseline_f1 is None:
                comparison.contention_comparison = ContentionComparison(
                    candidate_f1=candidate_f1, baseline_f1=None, regressed=False,
                    note="no baseline holdout evaluation available for the contention classifier",
                )
            else:
                any_comparable = True
                regressed = _regressed_higher_is_worse(candidate_f1, baseline_f1, regression_tolerance)
                if regressed:
                    any_regression = True
                    comparison.reasons.append(
                        f"contention_classifier: holdout F1 regressed beyond "
                        f"{regression_tolerance:.0%} tolerance versus active version {active_version}"
                    )
                comparison.contention_comparison = ContentionComparison(
                    candidate_f1=candidate_f1, baseline_f1=baseline_f1, regressed=regressed,
                )

    if active_version is None:
        comparison.recommendation = "no_active_baseline"
    elif any_regression:
        comparison.recommendation = "reject"
    else:
        comparison.recommendation = "deploy"
        if not any_comparable:
            comparison.reasons.append(
                "No comparable baseline evaluations found for any outcome -- "
                "recommending deploy by default, but this is a weak comparison."
            )

    logger.info(
        "Retraining candidate compared against active version",
        extra={
            "candidate_version": candidate_version, "active_version": active_version,
            "recommendation": comparison.recommendation, "reasons": comparison.reasons,
        },
    )
    return comparison
