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

# Bumped whenever a change to this class would make an artifact pickled
# by an older version behave incorrectly if loaded (e.g. F1: adding
# ``_train_means`` and switching score()/score_batch() to use it instead
# of a bare 0.0 fill). ``load_anomaly_detector`` rejects any artifact
# that doesn't carry this exact value rather than silently loading one
# that would score with the old, divergent imputation.
#
# The dataclass field's own default is deliberately *not* this value
# (see ``schema_version`` below) — a plain scalar dataclass default is
# also the *class* attribute, so an artifact pickled before this field
# existed would fall back to it on load (pickle restores __dict__, not
# __init__) and appear to already be current. Only ``fit()`` ever stamps
# an instance with the real version, so anything else reads as unversioned.
ANOMALY_ARTIFACT_SCHEMA_VERSION = 2
_UNVERSIONED = 0


@dataclass
class AnomalyDetector:
    """A trained (or trainable) Isolation Forest anomaly scorer."""

    params: dict = field(default_factory=lambda: dict(DEFAULT_ISOLATION_FOREST_PARAMS))
    min_training_rows: int = 20

    model: IsolationForest | None = field(default=None, repr=False)
    feature_columns: list[str] = field(default_factory=list)
    schema_version: int = _UNVERSIONED
    _train_means: pd.Series | None = field(default=None, repr=False)
    _train_score_mean: float | None = None
    _train_score_std: float | None = None

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """The one imputation path, used by ``fit``, ``score`` and ``score_batch`` alike.

        Reindexes to ``feature_columns`` and fills gaps from the
        *training* means (``_train_means``, computed once in ``fit``) —
        never a bare ``0.0``. Filling live NaNs with 0.0 while training
        NaNs were filled with the column mean (the pre-Phase-3 bug, F1)
        silently shifted every real sample's imputed values away from
        what the model was fit on, which is why every real sample used
        to score close to ``1.0``. A column that was entirely NaN in
        training has a NaN mean itself, so it still falls back to 0.0 —
        there is no other value to use for it.
        """
        aligned = X.reindex(columns=self.feature_columns)
        if self._train_means is not None:
            aligned = aligned.fillna(self._train_means)
        return aligned.fillna(0.0).astype(float)

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
        unlike LightGBM's. The same means are persisted on the artifact
        and reused by :meth:`score`/:meth:`score_batch` via
        :meth:`_prepare`, so scoring never diverges from what the model
        was fit on.
        """
        if len(X) < self.min_training_rows:
            raise AnomalyDetectionError(
                f"Not enough training rows: {len(X)} < {self.min_training_rows}",
                recoverable=True, context={"rows": len(X)},
            )

        self.feature_columns = list(X.columns)
        self._train_means = X.mean(numeric_only=True)
        filled = self._prepare(X)

        self.model = IsolationForest(**self.params)
        self.model.fit(filled)

        raw_scores = self.model.score_samples(filled)
        self._train_score_mean = float(np.mean(raw_scores))
        self._train_score_std = float(np.std(raw_scores)) or 1e-9  # avoid div-by-zero
        self.schema_version = ANOMALY_ARTIFACT_SCHEMA_VERSION

    def score(self, row: pd.Series) -> float:
        """Normalized anomaly score for one row: 0 (typical) to 1 (highly anomalous)."""
        if self.model is None:
            raise AnomalyDetectionError("score() called before fit()", recoverable=False)

        x = self._prepare(row.to_frame().T)
        raw_score = float(self.model.score_samples(x)[0])

        z = (self._train_score_mean - raw_score) / self._train_score_std
        return float(np.clip(z / _NORMALIZATION_SIGMA, 0.0, 1.0))

    def score_batch(self, X: pd.DataFrame) -> pd.Series:
        """Vectorized scoring over many rows — used by the training/evaluation pipeline."""
        if self.model is None:
            raise AnomalyDetectionError("score_batch() called before fit()", recoverable=False)

        aligned = self._prepare(X)
        raw_scores = self.model.score_samples(aligned)
        z = (self._train_score_mean - raw_scores) / self._train_score_std
        normalized = np.clip(z / _NORMALIZATION_SIGMA, 0.0, 1.0)
        return pd.Series(normalized, index=X.index)

    def is_significant(self, score: float, threshold: float) -> bool:
        return score >= threshold
