"""
The shared ``Forecaster`` interface and the three mandatory baselines
(Section 10.8): naive persistence, moving average, and a classical
statistical method.

Every forecaster — baseline or the real LightGBM model in
:mod:`aiperfopt.forecasting.model` — implements the same
``predict(row, history) -> ForecastResult`` signature. This is what
makes FR-5.3 ("independently swappable per metric without changing
downstream consumers' interface") concrete rather than aspirational: the
walk-forward evaluation harness (:mod:`aiperfopt.forecasting.evaluation`)
and Section 10.7's contingency plan (fall back to the best baseline if
LightGBM doesn't meaningfully outperform it) both just call
``forecaster.predict(...)`` without caring which implementation they hold.

``history`` is only used by :class:`ExponentialSmoothingForecaster` (it
needs the trailing raw series, not just one feature row) — every other
forecaster ignores it. Passing it uniformly to all of them, rather than
giving ETS a different method signature, is what keeps the interface
actually uniform.

Classical statistical baseline choice
----------------------------------------
Section 10.8 allows either ARIMA or Exponential Smoothing. Exponential
Smoothing (Holt's linear-trend method) is used here rather than ARIMA:
it's more robust to fit reliably on the short, noisy windows available
per evaluation point, requires no order-selection step, and per Section
3's stated principle ("if a simpler model performs comparably, the
simpler model is preferred, and the decision documented transparently"),
that's exactly the tradeoff being made here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from statsmodels.tsa.holtwinters import Holt

from aiperfopt.applog import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ForecastResult:
    predicted_value: float
    confidence: float | None
    model_version: str


class Forecaster(Protocol):
    model_version: str

    def predict(self, row: pd.Series, history: pd.Series | None = None) -> ForecastResult: ...


@dataclass
class NaivePersistenceForecaster:
    """Predicts "no change": the value ``horizon_seconds`` from now equals now."""

    model_version: str = "baseline-naive_persistence"

    def predict(self, row: pd.Series, history: pd.Series | None = None) -> ForecastResult:
        return ForecastResult(float(row["current_value"]), None, self.model_version)


@dataclass
class MovingAverageForecaster:
    """Predicts the trailing rolling mean, reusing Feature Engineering's own
    ``{metric}_roll_mean_{window}s`` column rather than recomputing it."""

    metric_name: str
    window_seconds: int
    model_version: str = "baseline-moving_average"

    def predict(self, row: pd.Series, history: pd.Series | None = None) -> ForecastResult:
        column = f"{self.metric_name}_roll_mean_{self.window_seconds}s"
        value = row.get(column)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            value = row["current_value"]  # no rolling window available yet — fall back
        return ForecastResult(float(value), None, self.model_version)


@dataclass
class ExponentialSmoothingForecaster:
    """Fits Holt's linear-trend method on the trailing raw window at each prediction.

    Capped to ``max_history_points`` for speed — fitting statsmodels'
    optimizer per evaluation row is the real cost driver of this
    baseline; V1 accepts this since walk-forward evaluation runs on
    bounded synthetic datasets, not months of real data (Section 10.3).
    """

    horizon_seconds: int
    max_history_points: int = 60
    model_version: str = "baseline-exponential_smoothing"

    def predict(self, row: pd.Series, history: pd.Series | None = None) -> ForecastResult:
        current_value = float(row["current_value"])

        if history is None:
            return ForecastResult(current_value, None, self.model_version)

        # Drop NaN *before* checking length/fitting — a trailing window
        # that crosses a scenario boundary (huge time jump between two
        # independent synthetic runs) or a counter reset can contain NaN
        # gaps. statsmodels' Holt does not reliably raise on NaN input;
        # it can silently propagate NaN through its computation and
        # return a NaN forecast without ever entering the except branch
        # below — found empirically while training on counter-rate
        # series that span multiple concatenated scenarios. Cleaning the
        # window first, and validating the output afterward, closes both
        # the entry point and the exit point for a silent NaN.
        clean_history = history.dropna()
        if len(clean_history) < 4:
            return ForecastResult(current_value, None, self.model_version)

        trimmed = clean_history.tail(self.max_history_points)
        try:
            fit = Holt(trimmed.to_numpy(), initialization_method="estimated").fit(optimized=True)
            elapsed = (trimmed.index[-1] - trimmed.index[0]).total_seconds()
            n = len(trimmed)
            avg_interval = elapsed / max(n - 1, 1) if n > 1 else 1.0
            steps = max(1, round(self.horizon_seconds / avg_interval)) if avg_interval > 0 else 1
            forecast = fit.forecast(steps)
            value = float(forecast[-1])
            if math.isnan(value) or math.isinf(value):
                raise ValueError("Holt forecast produced a non-finite value")
        except Exception as exc:  # noqa: BLE001 - statsmodels can fail on degenerate windows
            logger.debug("Exponential smoothing fit failed, falling back to current value",
                         extra={"error": str(exc)})
            value = current_value

        return ForecastResult(value, None, self.model_version)
