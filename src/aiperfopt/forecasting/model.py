"""
The trained forecaster (ADL-13: LightGBM per metric).

One :class:`MetricForecaster` instance covers exactly one (metric,
horizon) pair, wrapping exactly one LightGBM booster — this is what
ADL-06/Section 10.2 mean by "a single multi-step model per metric": one
model object per metric, not a family of hand-tuned separate pipelines
per horizon (the two horizons are two instances of the same, uniform
class, not two different architectures).

Confidence (FR-5.2)
----------------------
Confidence is set once per model, from that model's own walk-forward
validation RMSE relative to the target's historical spread — not
predicted per-input. This is deliberately the simpler of two reasonable
designs (the other being per-prediction quantile-regression intervals):
ADL-11 explicitly frames V1's confidence as "model-derived (uncalibrated)
... at lower cost," and Section 20's V1.5 roadmap describes statistical
calibration as a *future* enhancement layered on top of this — confirming
a simpler, validation-grounded scalar is the intended V1 scope, not a
shortcut around a harder requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import pandas as pd

from aiperfopt.common.exceptions import ForecastingError
from aiperfopt.features.config import FEATURE_SET_VERSION
from aiperfopt.forecasting.baselines import ForecastResult

# Sentinel distinct from any real ``FEATURE_SET_VERSION`` string — a
# forecaster fit before this field existed (or loaded from a metadata.json
# without it) reads as unversioned rather than silently matching the
# current version. Same dataclass-default trap as
# ``aiperfopt.anomaly.detector.AnomalyDetector.schema_version``: only
# ``fit()`` ever stamps the real value.
_UNVERSIONED_FEATURE_SET = ""

DEFAULT_LGBM_PARAMS: dict = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "learning_rate": 0.1,
    "feature_pre_filter": False,
}
DEFAULT_NUM_BOOST_ROUND = 80


@dataclass
class MetricForecaster:
    """A trained (or trainable) LightGBM forecaster for one metric/horizon pair."""

    metric_name: str
    horizon_label: str
    horizon_seconds: int
    model_version: str
    params: dict = field(default_factory=lambda: dict(DEFAULT_LGBM_PARAMS))
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND
    min_training_rows: int = 20

    booster: lgb.Booster | None = field(default=None, repr=False)
    feature_columns: list[str] = field(default_factory=list)
    feature_set_version: str = _UNVERSIONED_FEATURE_SET
    confidence: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the booster on a feature matrix ``X`` against target ``y``.

        ``X`` may contain NaN values (e.g. lag features across a gap —
        FR-4.3) — LightGBM natively handles missing values via its split
        -finding algorithm rather than requiring imputation, which is
        exactly why ADL-13 calls out "native interpretability" as part of
        the model choice; it extends to native NaN handling too, keeping
        Feature Engineering's explicit "leave it None" gap policy usable
        all the way through to training without a silent imputation step
        in between.
        """
        if len(X) < self.min_training_rows:
            raise ForecastingError(
                f"Not enough training rows for {self.metric_name}/{self.horizon_label}: "
                f"{len(X)} < {self.min_training_rows}",
                recoverable=True,
                context={"metric": self.metric_name, "horizon": self.horizon_label, "rows": len(X)},
            )

        self.feature_columns = list(X.columns)
        self.feature_set_version = FEATURE_SET_VERSION
        train_set = lgb.Dataset(X, label=y, free_raw_data=False)
        self.booster = lgb.train(self.params, train_set, num_boost_round=self.num_boost_round)

    def predict(self, row: pd.Series, history: pd.Series | None = None) -> ForecastResult:
        if self.booster is None:
            raise ForecastingError(
                "predict() called before fit()", recoverable=False,
                context={"metric": self.metric_name, "horizon": self.horizon_label},
            )
        x = row.reindex(self.feature_columns).to_frame().T.astype(float)
        predicted = float(self.booster.predict(x)[0])
        return ForecastResult(predicted, self.confidence, self.model_version)

    def predict_batch(self, X: pd.DataFrame) -> pd.Series:
        """Vectorized prediction over many rows at once — used by the evaluation harness."""
        if self.booster is None:
            raise ForecastingError("predict_batch() called before fit()", recoverable=False)
        aligned = X.reindex(columns=self.feature_columns).astype(float)
        return pd.Series(self.booster.predict(aligned), index=X.index)

    def set_confidence_from_validation(self, rmse: float, target_std: float) -> None:
        """Derive this model's confidence scalar from measured walk-forward RMSE.

        Bounded to [0, 1]: a validation RMSE of zero (perfect fit) gives
        confidence 1.0; an RMSE as large as the target's own historical
        standard deviation (i.e. the model is no better than always
        predicting the mean) gives confidence 0.0.
        """
        if target_std <= 0:
            # No variance in the validation target to normalize against —
            # a degenerate case (e.g. a perfectly flat synthetic scenario).
            self.confidence = 0.5
            return
        normalized_error = rmse / target_std
        self.confidence = float(max(0.0, min(1.0, 1.0 - normalized_error)))
