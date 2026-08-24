"""
Anomaly Detection (Section 7.1, item 7).

Isolation Forest on engineered features (Section 10.4): "a proper
unsupervised technique requiring no labels, moderate implementation
cost." Per the Section 7.2 data-flow diagram, this consumes the *same*
feature vectors Forecasting does — the full engineered feature matrix
from :mod:`aiperfopt.features`, not a narrower derived input like
Contention Prediction's forecast-plus-state vector.

Score convention
-------------------
scikit-learn's ``IsolationForest.score_samples()`` returns higher values
for *more normal* points (the opposite of what "anomaly score" usually
suggests colloquially). This module inverts and normalizes that into
:attr:`AnomalyDetector`'s own convention: a score in roughly ``[0, 1]``
where **higher means more anomalous**, calibrated against the *training
distribution's own* mean/std of raw scores (a z-score-style
normalization) rather than an arbitrary fixed scale — this is what makes
it "model-derived" rather than a heuristic pulled from nowhere,
consistent with this project's approach to confidence/scoring elsewhere
(see :mod:`aiperfopt.forecasting.model`'s docstring for the same
philosophy applied to forecast confidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from aiperfopt.common.exceptions import AnomalyDetectionError

DEFAULT_ISOLATION_FOREST_PARAMS: dict = {
    "n_estimators": 100,
    "contamination": "auto",
    "random_state": 42,
}

# How many standard deviations of the training score distribution map to
# a normalized anomaly score of 1.0. A point exactly at the training
# mean's raw score normalizes to 0.0; a point this many std-devs more
# "abnormal" than the mean normalizes to 1.0 (and is clipped beyond that,
# rather than growing unbounded).
_NORMALIZATION_SIGMA = 3.0


@dataclass
class AnomalyDetector:
    """A trained (or trainable) Isolation Forest anomaly scorer."""

    params: dict = field(default_factory=lambda: dict(DEFAULT_ISOLATION_FOREST_PARAMS))
    min_training_rows: int = 20

    model: IsolationForest | None = field(default=None, repr=False)
    feature_columns: list[str] = field(default_factory=list)
    _train_score_mean: float | None = None
    _train_score_std: float | None = None

    def fit(self, X: pd.DataFrame) -> None:
        """Train on a feature matrix representing typical operational data.

        ``X`` is expected to be Feature Engineering's full engineered
        matrix (or a concatenation of several scenarios' matrices, as
        :mod:`aiperfopt.contention.training_pipeline` does) — NaN values
        (gaps, per FR-4.3) are filled with each column's own mean before
        fitting, since scikit-learn's IsolationForest does not natively
        handle missing values the way LightGBM does. This is the one
        place in the codebase that imputes rather than passing NaN
        through, and it's a deliberate, narrow exception: Isolation
        Forest's splitting algorithm has no native missing-value path,
        unlike LightGBM's.
        """
        if len(X) < self.min_training_rows:
            raise AnomalyDetectionError(
                f"Not enough training rows: {len(X)} < {self.min_training_rows}",
                recoverable=True, context={"rows": len(X)},
            )

        self.feature_columns = list(X.columns)
        filled = X.fillna(X.mean(numeric_only=True))
        # Any column that was entirely NaN (mean itself NaN) falls back to 0.
        filled = filled.fillna(0.0)

        self.model = IsolationForest(**self.params)
        self.model.fit(filled)

        raw_scores = self.model.score_samples(filled)
        self._train_score_mean = float(np.mean(raw_scores))
        self._train_score_std = float(np.std(raw_scores)) or 1e-9  # avoid div-by-zero

    def score(self, row: pd.Series) -> float:
        """Normalized anomaly score for one row: 0 (typical) to 1 (highly anomalous)."""
        if self.model is None:
            raise AnomalyDetectionError("score() called before fit()", recoverable=False)

        x = row.reindex(self.feature_columns)
        x = x.fillna(0.0).to_frame().T.astype(float)
        raw_score = float(self.model.score_samples(x)[0])

        z = (self._train_score_mean - raw_score) / self._train_score_std
        return float(np.clip(z / _NORMALIZATION_SIGMA, 0.0, 1.0))

    def score_batch(self, X: pd.DataFrame) -> pd.Series:
        """Vectorized scoring over many rows — used by the training/evaluation pipeline."""
        if self.model is None:
            raise AnomalyDetectionError("score_batch() called before fit()", recoverable=False)

        aligned = X.reindex(columns=self.feature_columns).fillna(0.0).astype(float)
        raw_scores = self.model.score_samples(aligned)
        z = (self._train_score_mean - raw_scores) / self._train_score_std
        normalized = np.clip(z / _NORMALIZATION_SIGMA, 0.0, 1.0)
        return pd.Series(normalized, index=X.index)

    def is_significant(self, score: float, threshold: float) -> bool:
        return score >= threshold
