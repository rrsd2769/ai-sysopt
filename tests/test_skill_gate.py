"""Phase 6's quality gate (Section 11's precision-favoring stance, extended
from "don't regress vs. active" to "clear an absolute bar"). See
:mod:`aiperfopt.retraining.comparison`'s module-level docstring for how
this differs from ``compare_candidate_to_active``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score
from sqlalchemy import Engine

from aiperfopt.config.settings import ContentionSettings, ForecastingSettings, RetrainingSettings
from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.contention.training_pipeline import ContentionTrainingSummary
from aiperfopt.features import FeatureEngineeringConfig
from aiperfopt.forecasting.training_pipeline import MetricHorizonOutcome, TrainingRunSummary
from aiperfopt.retraining.comparison import evaluate_skill_gate
from aiperfopt.retraining.pipeline import run_retraining_on_real_usage_data
from aiperfopt.storage import insert_system_metric_samples_batch
from tests.factories import make_training_data

pytestmark = pytest.mark.behavioural

_MAE_MARGIN = 0.05
_DIRECTIONAL_MARGIN = 0.05
_F1_MARGIN = 0.05


def _fast_settings(tmp_path: Path):
    fconfig = FeatureEngineeringConfig(lookback_windows_seconds=(10, 30), tolerance_seconds=3.0)
    fsettings = ForecastingSettings(
        short_horizon_seconds=(10, 20), medium_horizon_seconds=(30, 60),
        walk_forward_splits=2, min_training_rows=15, models_dir=tmp_path / "models",
        directional_stability_threshold=0.02, baseline_improvement_margin=0.05,
    )
    csettings = ContentionSettings(
        decision_threshold=0.5, anomaly_dampening_factor=0.3,
        anomaly_significant_threshold=0.7, contributing_processes_top_n=3,
    )
    labeling_config = ContentionLabelingConfig(value_thresholds={"cpu_percent": 60.0}, tolerance_seconds=3.0)
    rsettings = RetrainingSettings(
        active_version_file=tmp_path / "active_model_version.json",
        regression_tolerance=0.10, real_usage_holdout_fraction=0.2,
        forecasting_mae_margin=_MAE_MARGIN, directional_accuracy_margin=_DIRECTIONAL_MARGIN,
        contention_f1_margin=_F1_MARGIN,
    )
    return fconfig, fsettings, csettings, labeling_config, rsettings


def _outcome(
    metric_name="cpu_percent", horizon_label="short",
    holdout_mae=1.0, holdout_persistence_mae=2.0, holdout_directional_accuracy=0.9,
):
    return MetricHorizonOutcome(
        metric_name=metric_name, horizon_label=horizon_label, horizon_seconds=45,
        walk_forward_summary={}, selected_forecaster="lightgbm",
        holdout_mae=holdout_mae, holdout_rmse=holdout_mae, holdout_directional_accuracy=holdout_directional_accuracy,
        holdout_persistence_mae=holdout_persistence_mae,
        holdout_persistence_directional_accuracy=0.5,
    )


def _contention_summary(candidate_f1=0.9, baseline_f1=0.5):
    return ContentionTrainingSummary(
        model_version="v1",
        holdout_metrics={"precision": candidate_f1, "recall": candidate_f1, "f1": candidate_f1},
        threshold_rule_baseline={"precision": baseline_f1, "recall": baseline_f1, "f1": baseline_f1},
    )


# --- Pure-function unit tests --------------------------------------------


def test_gate_passes_when_forecast_and_contention_both_clear_the_bar():
    forecast_summary = TrainingRunSummary(model_version="v1", outcomes=[_outcome()])
    gate = evaluate_skill_gate(
        forecast_summary, _contention_summary(), _MAE_MARGIN, _DIRECTIONAL_MARGIN, _F1_MARGIN
    )
    assert gate.passed is True
    assert gate.reasons == []
    assert gate.forecast_criteria[0].passed is True
    assert gate.contention_criterion.passed is True


def test_gate_fails_when_mae_improvement_is_below_margin():
    # 1.0 vs persistence 1.03 -- < 5% relative improvement.
    outcome = _outcome(holdout_mae=1.0, holdout_persistence_mae=1.03)
    forecast_summary = TrainingRunSummary(model_version="v1", outcomes=[outcome])
    gate = evaluate_skill_gate(forecast_summary, None, _MAE_MARGIN, _DIRECTIONAL_MARGIN, _F1_MARGIN)
    assert gate.passed is False
    assert gate.forecast_criteria[0].passed is False
    assert "MAE improvement" in gate.reasons[0]


def test_gate_fails_when_directional_accuracy_is_at_chance():
    outcome = _outcome(holdout_mae=0.5, holdout_persistence_mae=2.0, holdout_directional_accuracy=0.5)
    forecast_summary = TrainingRunSummary(model_version="v1", outcomes=[outcome])
    gate = evaluate_skill_gate(forecast_summary, None, _MAE_MARGIN, _DIRECTIONAL_MARGIN, _F1_MARGIN)
    assert gate.passed is False
    assert "directional accuracy" in gate.reasons[0]


def test_gate_fails_when_contention_f1_falls_far_short_of_baseline():
    forecast_summary = TrainingRunSummary(model_version="v1", outcomes=[_outcome()])
    gate = evaluate_skill_gate(
        forecast_summary, _contention_summary(candidate_f1=0.3, baseline_f1=0.5),
        _MAE_MARGIN, _DIRECTIONAL_MARGIN, _F1_MARGIN,
    )
    assert gate.passed is False
    assert gate.contention_criterion.passed is False
    assert any("contention_classifier" in r for r in gate.reasons)


def test_gate_treats_a_candidate_that_only_ties_a_perfect_baseline_as_passing():
    """A deterministic threshold rule can legitimately hit F1=1.0 on an easy
    holdout (e.g. a monotonic trend) -- nothing can *beat* perfect by a
    margin, and requiring one would fail every classifier that ties it.
    The gate's F1 criterion is a tolerance on how far the candidate may
    trail the baseline, not a required margin of superiority (see
    aiperfopt.retraining.comparison.evaluate_skill_gate), so tying --
    or even trailing by a little -- still passes."""
    forecast_summary = TrainingRunSummary(model_version="v1", outcomes=[_outcome()])
    gate = evaluate_skill_gate(
        forecast_summary, _contention_summary(candidate_f1=1.0, baseline_f1=1.0),
        _MAE_MARGIN, _DIRECTIONAL_MARGIN, _F1_MARGIN,
    )
    assert gate.contention_criterion.passed is True
    assert gate.passed is True


def test_gate_excludes_outcomes_with_no_holdout_evaluation_rather_than_failing_them():
    no_holdout = MetricHorizonOutcome(
        metric_name="mem_percent", horizon_label="short", horizon_seconds=45,
        walk_forward_summary={}, selected_forecaster="none",
        holdout_mae=None, holdout_rmse=None, holdout_directional_accuracy=None,
    )
    forecast_summary = TrainingRunSummary(model_version="v1", outcomes=[_outcome(), no_holdout])
    gate = evaluate_skill_gate(forecast_summary, None, _MAE_MARGIN, _DIRECTIONAL_MARGIN, _F1_MARGIN)
    assert gate.passed is True
    assert len(gate.forecast_criteria) == 1  # the excluded outcome contributed no criterion


def test_gate_passes_and_says_so_when_nothing_at_all_is_evaluable():
    """Mirrors Phase 5's honest single-class real-data result: no forecast
    outcome had holdout data and no contention classifier trained. Not a
    failure -- there's nothing to fail -- but not silent either."""
    forecast_summary = TrainingRunSummary(model_version="v1", outcomes=[])
    gate = evaluate_skill_gate(forecast_summary, None, _MAE_MARGIN, _DIRECTIONAL_MARGIN, _F1_MARGIN)
    assert gate.passed is True
    assert len(gate.reasons) == 1
    assert "could not verify anything" in gate.reasons[0]


def test_gate_ignores_a_missing_contention_summary_without_failing():
    forecast_summary = TrainingRunSummary(model_version="v1", outcomes=[_outcome()])
    gate = evaluate_skill_gate(forecast_summary, None, _MAE_MARGIN, _DIRECTIONAL_MARGIN, _F1_MARGIN)
    assert gate.passed is True
    assert gate.contention_criterion is None


# --- Integration: real, fixed-seed training -------------------------------


def _seed_usage_history(engine: Engine) -> None:
    scenarios, holdout = make_training_data(n_train=400, n_holdout=100)
    insert_system_metric_samples_batch(engine, [*scenarios["train"], *holdout])


def test_skill_gate_passes_on_a_fixed_seed_training_run(storage_engine: Engine, tmp_path: Path):
    """A real, deterministic (fixed-seed) training run's candidate clears
    the gate. The data has clean, learnable dynamics (periodic threshold
    crossings), so both the forecaster and the contention classifier
    should comfortably beat their baselines."""
    fconfig, fsettings, csettings, labeling_config, rsettings = _fast_settings(tmp_path)

    _seed_usage_history(storage_engine)
    candidate = run_retraining_on_real_usage_data(
        storage_engine, fconfig, fsettings, csettings, labeling_config, rsettings,
        new_model_version="gate-fixed-seed", target_metrics=["cpu_percent"],
    )

    assert candidate.contention_summary is not None
    assert candidate.contention_summary.threshold_rule_baseline is not None
    gate = candidate.skill_gate
    assert gate.passed is True, gate.reasons


def test_skill_gate_fails_a_classifier_no_better_than_chance(storage_engine: Engine, tmp_path: Path):
    """A crippled candidate must fail the gate. Rather than re-deriving
    the whole training pipeline with permuted training labels, this
    permutes the *holdout* probabilities relative to the true holdout
    labels on an otherwise-real, otherwise-passing candidate -- the
    evaluation-side equivalent of training on shuffled labels: both
    destroy any real correspondence between prediction and truth, so a
    classifier evaluated this way should score at chance regardless of
    what it actually learned, while the deterministic threshold-rule
    baseline (computed straight from the true labels) is untouched and
    keeps its real, legitimate F1.
    """
    fconfig, fsettings, csettings, labeling_config, rsettings = _fast_settings(tmp_path)

    _seed_usage_history(storage_engine)
    candidate = run_retraining_on_real_usage_data(
        storage_engine, fconfig, fsettings, csettings, labeling_config, rsettings,
        new_model_version="gate-crippled", target_metrics=["cpu_percent"],
    )
    summary = candidate.contention_summary
    assert summary is not None
    assert summary.holdout_probabilities and summary.holdout_labels and summary.threshold_rule_baseline
    assert candidate.skill_gate.passed is True  # sanity: the un-crippled candidate really does pass

    rng = np.random.default_rng(0)
    shuffled_probabilities = rng.permutation(summary.holdout_probabilities)
    predictions = (shuffled_probabilities >= summary.decision_threshold).astype(int)
    crippled_f1 = float(f1_score(summary.holdout_labels, predictions, zero_division=0))
    baseline_f1 = summary.threshold_rule_baseline["f1"]
    assert crippled_f1 < baseline_f1 + _F1_MARGIN  # the permutation must actually cripple it

    crippled_summary = replace(summary, holdout_metrics={**summary.holdout_metrics, "f1": crippled_f1})

    gate = evaluate_skill_gate(
        candidate.forecast_summary, crippled_summary, _MAE_MARGIN, _DIRECTIONAL_MARGIN, _F1_MARGIN
    )
    assert gate.passed is False
    assert any("contention_classifier" in r for r in gate.reasons)
