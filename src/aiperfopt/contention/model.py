"""
The Contention Prediction classifier (Section 10.4, FR-6.1-6.3).

Logistic regression, not a small GBM
----------------------------------------
Section 10.4 explicitly permits either "logistic regression or small
GBM." Logistic regression is used here deliberately: its coefficients
give a natural, free per-feature attribution — exactly what FR-6.2's
"list of contributing metrics" needs — without reaching for SHAP, which
is properly Diagnosis/XAI's tool (Phase 6, FR-8.1) and would blur the
boundary between these two subsystems if used here too. This is the same
"if a simpler model performs comparably, prefer the simpler model,
documented transparently" principle (Section 3) already applied to the
classical-statistics baseline choice in Forecasting.

Standardization
------------------
Logistic regression coefficients are only directly comparable across
features when those features share a common scale — otherwise a
feature's raw coefficient magnitude reflects its input's scale as much as
its actual importance (e.g. a byte-rate feature in the thousands would
dwarf a percentage feature in [0,100] even if the percentage feature
matters more). Inputs are standardized (zero mean, unit variance) before
fitting, and contribution scores are computed on the *standardized*
input, making the coefficients themselves a fair basis for ranking
"contributing metrics."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aiperfopt.common.exceptions import ContentionPredictionError

DEFAULT_LOGISTIC_REGRESSION_PARAMS: dict = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": 42,
}


@dataclass(frozen=True)
class ContributingMetric:
    metric_name: str
    contribution: float  # signed: positive pushes toward contention, negative away


@dataclass
class ContentionClassifier:
    """A trained (or trainable) logistic-regression contention classifier."""

    params: dict = field(default_factory=lambda: dict(DEFAULT_LOGISTIC_REGRESSION_PARAMS))
    min_training_rows: int = 30

    model: LogisticRegression | None = field(default=None, repr=False)
    scaler: StandardScaler | None = field(default=None, repr=False)
    feature_columns: list[str] = field(default_factory=list)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if len(X) < self.min_training_rows:
            raise ContentionPredictionError(
                f"Not enough training rows: {len(X)} < {self.min_training_rows}",
                recoverable=True, context={"rows": len(X)},
            )
        if y.nunique() < 2:
            raise ContentionPredictionError(
                "Training data contains only one class — cannot fit a classifier",
                recoverable=True, context={"unique_labels": y.unique().tolist()},
            )

        self.feature_columns = list(X.columns)
        filled = X.fillna(X.mean(numeric_only=True)).fillna(0.0)

        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(filled)

        self.model = LogisticRegression(**self.params)
        self.model.fit(scaled, y)

    def predict_proba(self, row: pd.Series) -> float:
        """Raw model probability, before anomaly-score dampening (FR-6.2)."""
        if self.model is None or self.scaler is None:
            raise ContentionPredictionError("predict_proba() called before fit()", recoverable=False)

        x = row.reindex(self.feature_columns).fillna(0.0).to_frame().T.astype(float)
        scaled = self.scaler.transform(x)
        return float(self.model.predict_proba(scaled)[0, 1])

    def predict_proba_batch(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None or self.scaler is None:
            raise ContentionPredictionError("predict_proba_batch() called before fit()", recoverable=False)

        aligned = X.reindex(columns=self.feature_columns).fillna(0.0).astype(float)
        scaled = self.scaler.transform(aligned)
        probabilities = self.model.predict_proba(scaled)[:, 1]
        return pd.Series(probabilities, index=X.index)

    def contributing_metrics(self, row: pd.Series, top_n: int = 5) -> list[ContributingMetric]:
        """Rank features by their (coefficient x standardized value) contribution
        to this specific prediction — FR-6.2's "list of contributing metrics."

        Ranked by absolute contribution so the top entries are whichever
        features moved *this* prediction most, in either direction, not
        just the features with the largest coefficients in general (a
        globally-important feature that happens to be near-average for
        this particular row contributed little to *this* prediction).
        """
        if self.model is None or self.scaler is None:
            raise ContentionPredictionError("contributing_metrics() called before fit()", recoverable=False)

        x = row.reindex(self.feature_columns).fillna(0.0).to_numpy(dtype=float)
        standardized = (x - self.scaler.mean_) / self.scaler.scale_
        contributions = standardized * self.model.coef_[0]

        ranked_indices = np.argsort(-np.abs(contributions))[:top_n]
        return [
            ContributingMetric(metric_name=self.feature_columns[i], contribution=float(contributions[i]))
            for i in ranked_indices
        ]
