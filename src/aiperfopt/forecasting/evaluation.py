"""
Model Evaluation & Validation (Section 7.1, item 10).

Walk-forward (rolling-origin) validation via ``sklearn.model_selection.
TimeSeriesSplit`` — the chronological-fold primitive Section 11/ADL-10
specifically calls for, as opposed to a random train/test split (which
would leak future information into training for time-series data).

Every fold trains a fresh :class:`~aiperfopt.forecasting.model.MetricForecaster`
on the fold's training rows and evaluates it, plus every baseline from
:mod:`aiperfopt.forecasting.baselines`, on the same held-out fold — so
model-vs-baseline comparisons are always apples-to-apples on identical
test data (FR-10.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from aiperfopt.applog import get_logger
from aiperfopt.common.timeutils import epoch_seconds_to_utc_datetime
from aiperfopt.forecasting.baselines import Forecaster, ForecastResult
from aiperfopt.forecasting.model import MetricForecaster

logger = get_logger(__name__)

_NON_FEATURE_COLUMNS = ("timestamp", "current_value", "target")


@dataclass(frozen=True)
class FoldMetrics:
    fold_index: int
    model_name: str
    mae: float
    rmse: float
    directional_accuracy: float
    n_test_rows: int


def feature_columns_of(dataset: pd.DataFrame) -> list[str]:
    return [c for c in dataset.columns if c not in _NON_FEATURE_COLUMNS]


def compute_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    errors = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    return mae, rmse


def compute_directional_accuracy(
    current: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, stability_threshold: float
) -> float:
    """Fraction of test points where the predicted direction (rising/falling/stable
    relative to ``current``) matches the actual direction (Section 11).

    A change is "stable" if it's within ``stability_threshold`` of
    ``current`` in relative terms — without this, two values that are
    both essentially flat but on opposite sides of exactly zero delta
    would be scored as a directional mismatch, which doesn't reflect
    anything meaningful about forecast quality.
    """

    def _direction(delta: float, reference: float) -> int:
        scale = max(abs(reference), 1e-9)
        if abs(delta) / scale < stability_threshold:
            return 0
        return 1 if delta > 0 else -1

    current = np.asarray(current, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(current) == 0:
        return float("nan")

    actual = [_direction(t - c, c) for t, c in zip(y_true, current)]
    predicted = [_direction(p - c, c) for p, c in zip(y_pred, current)]
    matches = sum(1 for a, p in zip(actual, predicted) if a == p)
    return matches / len(actual)


def build_history_lookup(raw_history: pd.Series | None):
    """Returns a function(timestamp) -> trailing history series up to that timestamp.

    Used only for :class:`~aiperfopt.forecasting.baselines.ExponentialSmoothingForecaster`,
    which needs the raw trailing window rather than one feature row.
    """
    if raw_history is None:
        return lambda ts: None

    def lookup(ts: float):
        cutoff = epoch_seconds_to_utc_datetime(pd.Series([ts])).iloc[0]
        return raw_history[raw_history.index <= cutoff]

    return lookup


def evaluate_forecaster_on_fold(
    forecaster: Forecaster,
    test_rows: pd.DataFrame,
    stability_threshold: float,
    raw_history: pd.Series | None = None,
) -> tuple[float, float, float]:
    """Evaluate one forecaster on one held-out fold. Returns (mae, rmse, directional_accuracy)."""
    history_lookup = build_history_lookup(raw_history)

    predictions = []
    for _, row in test_rows.iterrows():
        history = history_lookup(row["timestamp"])
        result: ForecastResult = forecaster.predict(row, history=history)
        predictions.append(result.predicted_value)

    mae, rmse = compute_mae_rmse(test_rows["target"].to_numpy(), np.array(predictions))
    directional = compute_directional_accuracy(
        test_rows["current_value"].to_numpy(), test_rows["target"].to_numpy(),
        np.array(predictions), stability_threshold,
    )
    return mae, rmse, directional


def run_walk_forward_evaluation(
    dataset: pd.DataFrame,
    metric_name: str,
    horizon_label: str,
    horizon_seconds: int,
    model_version: str,
    n_splits: int,
    stability_threshold: float,
    min_training_rows: int,
    raw_history: pd.Series | None = None,
    baseline_factories: dict[str, Callable[[], Forecaster]] | None = None,
) -> list[FoldMetrics]:
    """Run walk-forward CV for the LightGBM model and every baseline, fold by fold.

    ``baseline_factories`` produces a *fresh* baseline instance per fold
    (needed because :class:`MovingAverageForecaster` etc. are effectively
    stateless anyway, but keeping the pattern uniform with the LightGBM
    case — which genuinely must retrain per fold — avoids a subtle
    asymmetry in how folds are evaluated).
    """
    feature_cols = feature_columns_of(dataset)
    n_rows = len(dataset)
    if n_rows < min_training_rows * 2:
        logger.warning(
            "Not enough rows for walk-forward evaluation, skipping",
            extra={"metric": metric_name, "horizon": horizon_label, "rows": n_rows},
        )
        return []

    splitter = TimeSeriesSplit(n_splits=n_splits)
    results: list[FoldMetrics] = []

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(np.arange(n_rows))):
        if len(train_idx) < min_training_rows:
            continue
        train_rows = dataset.iloc[train_idx]
        test_rows = dataset.iloc[test_idx]

        model = MetricForecaster(
            metric_name=metric_name, horizon_label=horizon_label, horizon_seconds=horizon_seconds,
            model_version=model_version, min_training_rows=min_training_rows,
        )
        try:
            model.fit(train_rows[feature_cols], train_rows["target"])
        except Exception:  # noqa: BLE001
            logger.warning("Fold training failed, skipping fold", exc_info=True,
                            extra={"metric": metric_name, "horizon": horizon_label, "fold": fold_index})
            continue

        mae, rmse, directional = evaluate_forecaster_on_fold(model, test_rows, stability_threshold)
        results.append(FoldMetrics(fold_index, "lightgbm", mae, rmse, directional, len(test_rows)))

        for baseline_name, factory in (baseline_factories or {}).items():
            baseline = factory()
            b_mae, b_rmse, b_directional = evaluate_forecaster_on_fold(
                baseline, test_rows, stability_threshold, raw_history=raw_history
            )
            results.append(
                FoldMetrics(fold_index, baseline_name, b_mae, b_rmse, b_directional, len(test_rows))
            )

    return results


def mean_metrics_by_model(fold_results: list[FoldMetrics]) -> dict[str, dict[str, float]]:
    """Aggregate per-fold results into mean MAE/RMSE/directional accuracy per model name.

    Uses ``nanmean`` rather than ``mean`` deliberately: a single fold
    producing a non-finite metric (e.g. a forecaster hitting a genuinely
    degenerate input) should not silently poison the *entire* aggregate
    for that model across every other, perfectly good fold — ``np.mean``
    propagates a single NaN to the whole result, which would make one bad
    fold invisible as "the model's overall MAE is NaN" rather than
    surfacing as a specific, investigable fold-level warning. Every
    forecaster in this codebase is expected to never actually return
    NaN (see :class:`~aiperfopt.forecasting.baselines.ExponentialSmoothingForecaster`'s
    explicit non-finite check), so this is a defense-in-depth safety net,
    not a substitute for that guarantee.
    """
    by_model: dict[str, list[FoldMetrics]] = {}
    for result in fold_results:
        by_model.setdefault(result.model_name, []).append(result)

    summary = {}
    for name, rows in by_model.items():
        mae_values = [r.mae for r in rows]
        rmse_values = [r.rmse for r in rows]
        directional_values = [r.directional_accuracy for r in rows]
        if any(np.isnan(v) for v in mae_values + rmse_values):
            logger.warning(
                "Non-finite metric encountered in walk-forward folds",
                extra={"model_name": name, "mae_values": mae_values, "rmse_values": rmse_values},
            )
        summary[name] = {
            "mae": float(np.nanmean(mae_values)),
            "rmse": float(np.nanmean(rmse_values)),
            "directional_accuracy": float(np.nanmean(directional_values)),
            "n_folds": len(rows),
        }
    return summary


def select_best_forecaster_name(
    summary: dict[str, dict[str, float]], improvement_margin: float
) -> str:
    """Section 10.7's contingency plan: pick "lightgbm" only if it beats every
    baseline's mean MAE by at least ``improvement_margin`` (relative); otherwise
    fall back to whichever baseline had the lowest mean MAE.

    A model that does not meaningfully outperform the baselines is not
    deployed as the primary forecaster (Section 10.8) — this function is
    the concrete decision point that enforces that rule, and its outcome
    is always stored (see :mod:`aiperfopt.forecasting.training_pipeline`)
    so the choice is documented transparently either way (ADL-35).
    """
    baseline_names = [name for name in summary if name != "lightgbm"]
    if not baseline_names:
        return "lightgbm"

    best_baseline_name = min(baseline_names, key=lambda name: summary[name]["mae"])
    best_baseline_mae = summary[best_baseline_name]["mae"]

    if "lightgbm" not in summary:
        return best_baseline_name

    lightgbm_mae = summary["lightgbm"]["mae"]
    if best_baseline_mae <= 0:
        return "lightgbm" if lightgbm_mae <= best_baseline_mae else best_baseline_name

    improvement = (best_baseline_mae - lightgbm_mae) / best_baseline_mae
    return "lightgbm" if improvement >= improvement_margin else best_baseline_name
