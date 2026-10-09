"""
Evaluation for the Contention Prediction classifier.

The specific metrics (MAE, RMSE, directional accuracy) are defined
for *forecasting* models — they don't apply to a binary classifier.
The broader methodology (chronological walk-forward folds,
precision favored over recall since false positives erode trust faster
than occasional misses) does apply, and is what this module implements:
``TimeSeriesSplit`` chronological folds (consistent with
:mod:`aiperfopt.forecasting.evaluation`), precision/recall/F1 at the
configured decision threshold. Results are persisted into the same
``evaluations`` table Forecasting uses, with the classification-specific
metrics carried in the nullable ``extra`` JSON column rather than the
MAE/RMSE/directional_accuracy columns (which are left null for these
rows) — no schema change needed, since the schema already
anticipated evaluation rows not populating every numeric column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import ContentionPredictionError
from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.contention.model import ContentionClassifier

logger = get_logger(__name__)

_NON_FEATURE_COLUMNS = ("timestamp", "label")

_DEFAULT_DECISION_THRESHOLD = 0.5  # fallback when a PR curve can't be fit (see fit_decision_threshold)


@dataclass(frozen=True)
class ContentionFoldMetrics:
    fold_index: int
    precision: float
    recall: float
    f1: float
    pr_auc: float
    n_test_rows: int
    n_positive: int


def compute_base_rate(labels: pd.Series) -> float:
    """Fraction of positive labels — the number an accuracy figure would
    hide (report the base rate explicitly: real passive
    data is heavily negative, so even a model that always predicts "no
    contention" scores a deceptively high accuracy)."""
    if len(labels) == 0:
        return float("nan")
    return float(pd.Series(labels).astype(float).mean())


def compute_pr_auc(probabilities: pd.Series, labels: pd.Series) -> float:
    """Average precision (area under the precision-recall curve).

    ``nan`` when ``labels`` is single-class -- PR-AUC is undefined
    without both classes present, and a fold/holdout landing on all-one-
    class should surface as "not computable" rather than a misleading
    number.
    """
    if pd.Series(labels).nunique() < 2:
        return float("nan")
    return float(average_precision_score(labels, probabilities))


def fit_decision_threshold(probabilities: pd.Series, labels: pd.Series) -> tuple[float, dict[str, float]]:
    """Pick the decision threshold from a precision-recall curve rather
    than a config constant (the operational parameter
    now has a data-driven starting point): the threshold on the curve
    that maximizes F1.

    Falls back to :data:`_DEFAULT_DECISION_THRESHOLD` when the curve is
    degenerate (single-class labels, or no threshold achieves a non-zero
    F1) -- an arbitrary threshold on an unusable curve is no better than
    the old constant, so this is a clearly-labeled fallback, not a
    disguised failure.

    Returns ``(threshold, metrics)`` where ``metrics`` is the
    precision/recall/f1 achieved *at that threshold* on the same data —
    stored alongside the threshold so a report can show the curve point
    that was actually chosen, not just the number.
    """
    if pd.Series(labels).nunique() < 2:
        return _DEFAULT_DECISION_THRESHOLD, {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    precisions, recalls, thresholds = precision_recall_curve(labels, probabilities)
    # precision_recall_curve returns one more precision/recall point than
    # threshold (the last point is recall=0, precision=1 with no
    # corresponding threshold) -- drop it so arrays line up.
    precisions, recalls = precisions[:-1], recalls[:-1]
    if len(thresholds) == 0:
        return _DEFAULT_DECISION_THRESHOLD, {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = np.where(
            (precisions + recalls) > 0, 2 * precisions * recalls / (precisions + recalls), 0.0
        )
    if not np.any(f1s > 0):
        return _DEFAULT_DECISION_THRESHOLD, {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), {
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "f1": float(f1s[best_idx]),
    }


def evaluate_threshold_rule_baseline(
    dataset: pd.DataFrame, labeling_config: ContentionLabelingConfig
) -> dict[str, float]:
    """The baseline contention prediction never had: predict
    contention iff the *current* value of any tracked metric already
    meets its own labeling threshold -- no model, no forecast, just "is
    it already bad right now."

    Reads whichever ``current_{metric}`` columns are present in
    ``dataset`` (the same shape :func:`aiperfopt.contention.dataset.
    build_contention_dataset` produces) and compares each against
    ``labeling_config.value_thresholds``. A deterministic rule has no
    probability to rank by, so only precision/recall/f1 are reported —
    PR-AUC does not apply to a non-scoring baseline.
    """
    predictions = pd.Series(0, index=dataset.index)
    for metric_name, threshold in labeling_config.value_thresholds.items():
        column = f"current_{metric_name}"
        if column not in dataset.columns:
            continue
        predictions = predictions | (dataset[column] >= threshold).astype(int)

    labels = dataset["label"]
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


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
        pr_auc = compute_pr_auc(probabilities, test_rows["label"])

        results.append(
            ContentionFoldMetrics(
                fold_index=fold_index, precision=float(precision), recall=float(recall), f1=float(f1),
                pr_auc=pr_auc, n_test_rows=len(test_rows), n_positive=int(test_rows["label"].sum()),
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
        "pr_auc": float(np.nanmean([r.pr_auc for r in fold_results])),
        "n_folds": float(len(fold_results)),
    }
