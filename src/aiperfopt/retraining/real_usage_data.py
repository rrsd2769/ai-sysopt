"""
Turns accumulated real-usage history (Section 10.3 Phase 2) into the same
``scenarios`` / ``holdout_samples`` shape Forecasting and Contention's
training pipelines already accept for synthetic data (Phase 4/5), so
retraining against real data needs no separate dataset-construction logic
of its own — see :mod:`aiperfopt.forecasting.training_pipeline` and
:mod:`aiperfopt.contention.training_pipeline`'s new ``scenarios``/
``holdout_samples`` parameters.

Why a single chronological split, not synthetic's multi-scenario library
--------------------------------------------------------------------------
Phase 4's synthetic bootstrap combines several *independent, deliberately
varied* scenarios (idle, cpu_stress, ...) because walk-forward CV needs
shape variety to generalize (see that module's docstring). Real-usage
data has no such artificial scenario boundaries — it is one continuous
recording of whatever the user's machine actually did. So this module
does not try to invent scenario splits; it produces exactly one
"scenario" (the training portion of history) plus one held-out slice,
consistent with Section 11's requirement that the held-out check use the
*most recent* data, never an earlier slice, to avoid future-to-past
leakage (ADL-10) — the opposite of a random split.
"""

from __future__ import annotations

from sqlalchemy import Engine

from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import RetrainingError
from aiperfopt.storage.reader import get_system_metrics_range
from aiperfopt.storage.records import SystemMetricSample

logger = get_logger(__name__)

# The scenario name real-usage training data is filed under — mirrors
# how synthetic scenarios each get a name (Section 10.3 Phase 1), so
# downstream code (feature computation, logging) has something
# meaningful to attribute this data to.
REAL_USAGE_SCENARIO_NAME = "real_usage"


def build_real_usage_training_data(
    engine: Engine,
    start: float | None = None,
    end: float | None = None,
    holdout_fraction: float = 0.2,
    min_total_rows: int = 1,
) -> tuple[dict[str, list[SystemMetricSample]], list[SystemMetricSample]]:
    """Read accumulated real-usage history and split it chronologically.

    Parameters
    ----------
    start / end:
        Optional bounds on which stored history to use. Defaults to all
        available history when omitted.
    holdout_fraction:
        Fraction of the chronologically-ordered rows reserved as the
        held-out generalization check (Section 11) — always the *most
        recent* fraction, never an earlier slice, so training data never
        contains information from after the holdout period (ADL-10).
    min_total_rows:
        Minimum number of rows required across the whole window before
        this is considered usable retraining data. Real usage data
        accumulates gradually (Section 10.3: "not required to mature
        within the course timeline") — calling this too early should
        fail clearly rather than silently training on a handful of rows.

    Returns
    -------
    A ``(scenarios, holdout_samples)`` pair in exactly the shape
    :func:`aiperfopt.forecasting.training_pipeline.train_and_evaluate_all`
    and
    :func:`aiperfopt.contention.training_pipeline.train_contention_and_anomaly_pipeline`
    accept via their ``scenarios``/``holdout_samples`` parameters.
    ``scenarios`` always has exactly one key, ``REAL_USAGE_SCENARIO_NAME``.

    Raises
    ------
    RetrainingError
        If fewer than ``min_total_rows`` samples are available, or if the
        chronological split would leave either side empty.
    """
    samples = get_system_metrics_range(engine, start=start, end=end)

    if len(samples) < min_total_rows:
        raise RetrainingError(
            f"Not enough accumulated real-usage data to retrain: "
            f"{len(samples)} rows available, {min_total_rows} required",
            recoverable=True,
            context={"rows_available": len(samples), "rows_required": min_total_rows},
        )

    split_index = int(len(samples) * (1.0 - holdout_fraction))
    training_samples = samples[:split_index]
    holdout_samples = samples[split_index:]

    if not training_samples or not holdout_samples:
        raise RetrainingError(
            "Chronological train/holdout split produced an empty side — "
            "not enough real-usage data yet for the configured holdout_fraction",
            recoverable=True,
            context={
                "total_rows": len(samples),
                "holdout_fraction": holdout_fraction,
                "training_rows": len(training_samples),
                "holdout_rows": len(holdout_samples),
            },
        )

    logger.info(
        "Built real-usage training data",
        extra={
            "training_rows": len(training_samples),
            "holdout_rows": len(holdout_samples),
            "start": training_samples[0].timestamp,
            "end": holdout_samples[-1].timestamp,
        },
    )

    return {REAL_USAGE_SCENARIO_NAME: training_samples}, holdout_samples
