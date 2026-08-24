from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aiperfopt.common.exceptions import ContentionPredictionError
from aiperfopt.contention.model import ContentionClassifier
from aiperfopt.contention.persistence import (
    contention_artifact_exists,
    load_contention_classifier,
    save_contention_classifier,
)


def _trained_classifier() -> ContentionClassifier:
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 100, 60)
    X = pd.DataFrame({"feature_a": a, "feature_b": rng.uniform(0, 100, 60)})
    y = pd.Series((a > 60).astype(int))
    classifier = ContentionClassifier(min_training_rows=20)
    classifier.fit(X, y)
    return classifier


def test_save_and_load_round_trip(tmp_path: Path):
    classifier = _trained_classifier()
    save_contention_classifier(classifier, tmp_path, "v1")
    assert contention_artifact_exists(tmp_path, "v1") is True

    loaded = load_contention_classifier(tmp_path, "v1")
    row = pd.Series({"feature_a": 75.0, "feature_b": 20.0})
    assert loaded.predict_proba(row) == pytest.approx(classifier.predict_proba(row))


def test_save_untrained_raises(tmp_path: Path):
    with pytest.raises(ContentionPredictionError):
        save_contention_classifier(ContentionClassifier(), tmp_path, "v1")


def test_load_missing_raises(tmp_path: Path):
    with pytest.raises(ContentionPredictionError):
        load_contention_classifier(tmp_path, "does_not_exist")
