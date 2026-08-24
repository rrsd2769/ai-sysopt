"""
Load or reconstruct whichever forecaster Section 10.7's contingency plan
actually selected for a given metric/horizon.

:func:`aiperfopt.forecasting.training_pipeline.train_evaluate_and_select_one`
records *which* forecaster won (``MetricHorizonOutcome.selected_forecaster``)
and, only if LightGBM won, saves a loadable artifact — a baseline
"winning" has nothing to load, since baselines are parameterless/stateless
and can be reconstructed on demand. This module is the one place that
knows how to turn "the outcome of a training run" back into "a live
``Forecaster`` instance ready to call ``.predict()`` on" — Contention
Prediction (Phase 5, FR-6.1: consumes forecast *outputs*) is the first
real caller, but any future consumer of a deployed forecaster (the API
layer in Phase 9, for instance) should go through this rather than
re-deriving the load-or-reconstruct logic itself.

Reconstructing "which forecaster won" outside of a training run
------------------------------------------------------------------
``load_deployed_forecaster`` needs a full ``MetricHorizonOutcome``, which
only exists in memory immediately after
:func:`aiperfopt.forecasting.training_pipeline.train_and_evaluate_all`
returns. Phase 9's live-inference loop is the first caller that needs to
reconstruct deployed forecasters *without* holding onto that in-memory
object — e.g. after a process restart, or simply because the active
version was deployed in a different process than the one running live
inference. :func:`resolve_deployed_forecasters` rebuilds the outcome
metadata it needs from what's actually persisted: ``horizon_seconds`` and
``is_counter`` are deterministic from config (not training-run-specific
data), and ``selected_forecaster`` is recovered from the holdout
evaluation row Phase 4/8 already writes for every (metric, horizon) —
see :func:`aiperfopt.forecasting.training_pipeline.train_and_evaluate_all`
and :func:`aiperfopt.storage.reader.get_evaluations_by_version`.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine

from aiperfopt.config.settings import ForecastingSettings
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.forecasting.baselines import Forecaster
from aiperfopt.forecasting.persistence import load_forecaster
from aiperfopt.forecasting.training_pipeline import MetricHorizonOutcome, baseline_factories_for
from aiperfopt.storage.reader import get_evaluations_by_version


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
    that produced no usable model during that training run (Section
    10.7's contingency plan producing no candidate at all is rare but
    possible for a very short/sparse training window). A caller with
    zero usable forecasters back should treat that the same as "no
    active model" (NFR-11.1's graceful degradation), not raise.
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
