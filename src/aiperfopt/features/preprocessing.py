"""
Preprocessing between Feature Engineering and a model.

Feature Engineering (:mod:`aiperfopt.features.engine`) already does the
*per-sample* cleaning every row needs regardless of what consumes it: it
sorts and de-duplicates timestamps, turns cumulative OS counters into
per-second rates (treating resets and large gaps as missing), and leaves a
lag as NaN rather than reaching across a gap. This module adds the steps that
depend on the *dataset as a whole*, and so have to happen after the feature
matrix exists:

Row-level (stateless, :func:`clean_supervised_rows`)
    1. **Validity.** Drop rows whose current value or target is non-finite or
       outside the metric's physical range (0-100 for a percentage).
    2. **Warm-up.** Split the timeline into contiguous segments wherever two
       samples are more than ``gap_seconds`` apart (collector stopped, laptop
       asleep), and drop rows less than ``warmup_seconds`` into a segment.
       Their rolling windows cover less history than their names claim --
       a "120 s mean" two seconds after a restart is a mean of one sample.

Column-level (learned, :class:`FeaturePreprocessor`)
    3. **Log transform.** ``sign(x) * log1p(|x|)`` on every feature derived
       from a byte counter. Byte rates are extremely right-skewed (a download
       is orders of magnitude above idle); on a log scale a linear model can
       use them and a tree's histogram bins are spread over the real range.
    4. **Constant features.** Dropped -- they carry no information.
    5. **Redundant features.** Of any pair correlated above
       ``redundancy_threshold``, the later column is dropped. ``current_value``
       is never dropped: the delta-target reconstruction needs it.

Steps 4-5 are *fitted*: which columns to keep is decided from the training
rows only and then applied unchanged to validation/holdout rows, so no
information from evaluation data leaks into the choice. Missing values are
deliberately left as NaN -- tree models split on them natively, and a model
that can't (Ridge) imputes inside its own pipeline, also fitted on training
rows only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aiperfopt.features.config import FeatureEngineeringConfig

ALWAYS_KEEP = ("current_value",)
PERCENT_RANGE = (0.0, 100.0)


@dataclass
class RowCleaningReport:
    input_rows: int
    invalid_rows: int
    warmup_rows: int
    segments: int
    output_rows: int


def segment_ids(timestamps: pd.Series, gap_seconds: float) -> pd.Series:
    """Contiguous-segment id per row: increments wherever the time step exceeds ``gap_seconds``."""
    ts = pd.Series(np.asarray(timestamps, dtype=float), index=timestamps.index)
    return (ts.diff() > gap_seconds).cumsum()


def clean_supervised_rows(
    dataset: pd.DataFrame,
    *,
    gap_seconds: float,
    warmup_seconds: float,
    value_range: tuple[float, float] | None = PERCENT_RANGE,
) -> tuple[pd.DataFrame, RowCleaningReport]:
    """Steps 1-2: drop invalid rows and rows inside a segment's warm-up period.

    ``dataset`` is a supervised dataset from
    :func:`aiperfopt.forecasting.dataset.build_supervised_dataset` (it needs
    ``timestamp``, ``current_value`` and ``target``). Segments are computed on
    the full, unfiltered timeline, so a dropped invalid row never creates or
    hides a gap.
    """
    ts = dataset["timestamp"].astype(float)
    seg = segment_ids(ts, gap_seconds)
    since_start = ts - ts.groupby(seg).transform("min")
    warm = since_start < warmup_seconds

    values = dataset[["current_value", "target"]].astype(float)
    invalid = ~np.isfinite(values).all(axis=1)
    if value_range is not None:
        lo, hi = value_range
        invalid |= ((values < lo) | (values > hi)).any(axis=1)

    keep = ~(invalid | warm)
    report = RowCleaningReport(
        input_rows=len(dataset),
        invalid_rows=int(invalid.sum()),
        warmup_rows=int((warm & ~invalid).sum()),
        segments=int(seg.nunique()),
        output_rows=int(keep.sum()),
    )
    return dataset.loc[keep].reset_index(drop=True), report


def is_counter_derived(column: str, counter_metrics: tuple[str, ...]) -> bool:
    return any(column.startswith(f"{metric}_rate") for metric in counter_metrics)


def signed_log1p(values: pd.DataFrame) -> pd.DataFrame:
    return np.sign(values) * np.log1p(values.abs())


@dataclass
class FeaturePreprocessor:
    """Steps 3-5, fitted on training rows and reused unchanged on any later rows."""

    redundancy_threshold: float = 0.99
    counter_metrics: tuple[str, ...] = field(default_factory=lambda: FeatureEngineeringConfig().counter_metrics)
    log_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    redundant_columns: list[str] = field(default_factory=list)
    kept_columns: list[str] = field(default_factory=list)
    input_columns: list[str] = field(default_factory=list)

    def fit(self, X: pd.DataFrame) -> FeaturePreprocessor:
        self.input_columns = list(X.columns)
        self.log_columns = [c for c in X.columns if is_counter_derived(c, self.counter_metrics)]
        Xt = self._log(X.astype(float))

        self.constant_columns = [
            c for c in Xt.columns if c not in ALWAYS_KEEP and Xt[c].nunique(dropna=True) <= 1
        ]
        candidates = [c for c in Xt.columns if c not in self.constant_columns]

        corr = Xt[candidates].corr().abs().to_numpy()
        np.fill_diagonal(corr, 0.0)
        dropped: set[int] = set()
        for i, name in enumerate(candidates):
            if i in dropped:
                continue
            for j in range(i + 1, len(candidates)):
                if j not in dropped and candidates[j] not in ALWAYS_KEEP and corr[i, j] > self.redundancy_threshold:
                    dropped.add(j)
        self.redundant_columns = [candidates[j] for j in sorted(dropped)]
        self.kept_columns = [c for k, c in enumerate(candidates) if k not in dropped]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.kept_columns:
            raise RuntimeError("FeaturePreprocessor.transform() called before fit()")
        Xt = self._log(X.reindex(columns=self.input_columns).astype(float))
        return Xt[self.kept_columns]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def _log(self, X: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self.log_columns if c in X.columns]
        if present:
            X = X.copy()
            X[present] = signed_log1p(X[present])
        return X
