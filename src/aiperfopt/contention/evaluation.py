"""
Evaluation for the Contention Prediction classifier.

FR-10.1's specific metrics (MAE, RMSE, directional accuracy) are defined
for *forecasting* models — they don't apply to a binary classifier.
Section 11's broader methodology (chronological walk-forward folds,
precision favored over recall since false positives erode trust faster
than occasional misses) does apply, and is what this module implements:
``TimeSeriesSplit`` chronological folds (consistent with
:mod:`aiperfopt.forecasting.evaluation`), precision/recall/F1 at the
configured decision threshold. Results are persisted into the same
``evaluations`` table Forecasting uses, with the classification-specific
metrics carried in the nullable ``extra`` JSON column rather than the
MAE/RMSE/directional_accuracy columns (which are left null for these
rows) — no schema change needed, since Phase 1's schema already
anticipated evaluation rows not populating every numeric column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import ContentionPredictionError
from aiperfopt.contention.model import ContentionClassifier

logger = get_logger(__name__)

_NON_FEATURE_COLUMNS = ("timestamp", "label")


@dataclass(frozen=True)
class ContentionFoldMetrics:
    fold_index: int
    precision: float
    recall: float
    f1: float
    n_test_rows: int
    n_positive: int


def feature_columns_of(dataset: pd.DataFrame) -> list[str]:
    return [c for c in dataset.columns if c not in _NON_FEATURE_COLUMNS]


def run_contention_walk_forward_evaluation(
    dataset: pd.DataFrame, decision_threshold: float, n_splits: int, min_training_rows: int
) -> list[ContentionFoldMetrics]:
    feature_cols = feature_columns_of(dataset)
    n_rows = len(dataset)
    if n_rows < min_training_rows * 2:
        logger.warning(
            "Not enough rows for contention walk-forward evaluation, skipping",
            extra={"rows": n_rows},
        )
        return []

    splitter = TimeSeriesSplit(n_splits=n_splits)
    results: list[ContentionFoldMetrics] = []

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(np.arange(n_rows))):
        if len(train_idx) < min_training_rows:
            continue
        train_rows = dataset.iloc[train_idx]
        test_rows = dataset.iloc[test_idx]

        if train_rows["label"].nunique() < 2:
            logger.debug("Skipping fold with single-class training data", extra={"fold": fold_index})
            continue

        classifier = ContentionClassifier(min_training_rows=min_training_rows)
        try:
            classifier.fit(train_rows[feature_cols], train_rows["label"])
        except ContentionPredictionError:
            logger.warning("Fold training failed, skipping fold", exc_info=True, extra={"fold": fold_index})
            continue

        probabilities = classifier.predict_proba_batch(test_rows[feature_cols])
        predictions = (probabilities >= decision_threshold).astype(int)

        precision = precision_score(test_rows["label"], predictions, zero_division=0)
        recall = recall_score(test_rows["label"], predictions, zero_division=0)
        f1 = f1_score(test_rows["label"], predictions, zero_division=0)

        results.append(
            ContentionFoldMetrics(
                fold_index=fold_index, precision=float(precision), recall=float(recall), f1=float(f1),
                n_test_rows=len(test_rows), n_positive=int(test_rows["label"].sum()),
            )
        )

    return results


def mean_contention_metrics(fold_results: list[ContentionFoldMetrics]) -> dict[str, float]:
    if not fold_results:
        return {}
    return {
        "precision": float(np.nanmean([r.precision for r in fold_results])),
        "recall": float(np.nanmean([r.recall for r in fold_results])),
        "f1": float(np.nanmean([r.f1 for r in fold_results])),
        "n_folds": float(len(fold_results)),
    }
