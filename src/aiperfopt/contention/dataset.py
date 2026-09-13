"""
Supervised dataset construction for Contention Prediction (FR-6.1).

FR-6.1 requires the classifier to consume "forecast outputs plus current
system state" — explicitly *not* raw/engineered metrics directly. This
module is what enforces that boundary structurally: it builds a dataset
whose only feature columns are ``forecast_{metric}`` (from each target
metric's already-trained, already-selected Phase 4 forecaster),
``current_{metric}`` (the metric's own present value), and
``anomaly_score`` (from a trained :class:`~aiperfopt.anomaly.detector.AnomalyDetector`
on the same feature vectors, per Section 7.2's data-flow diagram). The
full lag/rolling/rate engineered feature matrix from
:mod:`aiperfopt.features` is used only as an *intermediate* — to compute
forecasts and the anomaly score — and never appears directly in the
returned dataset's columns.

Labels
--------
A row is labeled "contention" (1) if *any* tracked metric's actual future
value (looked up the same way Forecasting's own training labels are —
see :mod:`aiperfopt.forecasting.dataset`) meets or exceeds its configured
threshold in :class:`~aiperfopt.contention.config.ContentionLabelingConfig`.
This reuses Forecasting's exact forward-looking lookup machinery (through
:func:`aiperfopt.forecasting.dataset.build_supervised_dataset`) rather
than a second, parallel implementation of "look into the future" — see
that module's docstring for why a second implementation would be a real
risk of a future-information-leakage bug.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from aiperfopt.anomaly.detector import AnomalyDetector
from aiperfopt.common.timeutils import epoch_seconds_to_utc_datetime
from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.features.engine import compute_feature_matrix, counter_rate_per_second
from aiperfopt.forecasting.baselines import Forecaster
from aiperfopt.forecasting.dataset import build_supervised_dataset
from aiperfopt.forecasting.evaluation import build_history_lookup
from aiperfopt.storage.records import FeatureVectorRecord, SystemMetricSample


def build_contention_dataset(
    samples: Sequence[SystemMetricSample],
    deployed_forecasters: dict[str, Forecaster],
    anomaly_detector: AnomalyDetector,
    feature_config: FeatureEngineeringConfig,
    labeling_config: ContentionLabelingConfig,
    horizon_seconds: int,
    process_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one scenario's contention training/evaluation rows.

    Returns a DataFrame with columns: ``timestamp``, one
    ``forecast_{metric}`` and ``current_{metric}`` pair per entry in
    ``deployed_forecasters``, ``anomaly_score``, and ``label`` (0/1).
    Rows are dropped if any target metric lacks a valid forward-looking
    label within tolerance (consistent with how Forecasting's own
    dataset construction handles the tail of a series).

    ``process_summary``: optional, already-aligned process-summary frame
    (Phase 4/5) merged into the intermediate engineered feature matrix
    used to compute the anomaly score -- see
    :func:`aiperfopt.features.engine.compute_feature_matrix`. Omitted
    (the default) reproduces exactly pre-Phase-5 behavior.
    """
    fm = compute_feature_matrix(samples, feature_config, process_summary=process_summary)
    if fm.empty or not deployed_forecasters:
        return pd.DataFrame()

    feature_cols = [c for c in fm.columns if c != "timestamp"]

    anomaly_scores = anomaly_detector.score_batch(fm[feature_cols])
    combined = pd.DataFrame({"timestamp": fm["timestamp"].to_numpy(), "anomaly_score": anomaly_scores.to_numpy()})

    for metric_name, forecaster in deployed_forecasters.items():
        is_counter = metric_name in feature_config.counter_metrics

        labeled = build_supervised_dataset(
            samples, fm, metric_name, horizon_seconds, labeling_config.tolerance_seconds,
            is_counter=is_counter,
        )
        if labeled.empty:
            return pd.DataFrame()  # this metric never has a valid label -> no usable rows at all

        # Build real trailing history for forecasters that need it (e.g.
        # ExponentialSmoothingForecaster) — without this, every forecaster
        # would silently degrade to its history=None fallback (effectively
        # naive persistence), understating what the actually-deployed
        # forecaster (which may genuinely be ExponentialSmoothingForecaster
        # — Section 10.7's contingency plan can and does select it) would
        # really produce.
        raw_df = pd.DataFrame.from_records(
            [{"timestamp": s.timestamp, metric_name: getattr(s, metric_name)} for s in samples]
        )
        raw_df = raw_df.dropna(subset=[metric_name]).sort_values("timestamp")
        raw_df["_dt"] = epoch_seconds_to_utc_datetime(raw_df["timestamp"])
        raw_series = raw_df.set_index("_dt")[metric_name].astype(float)
        if is_counter:
            raw_series = counter_rate_per_second(raw_series, max_gap_seconds=30.0)
        history_lookup = build_history_lookup(raw_series)

        forecast_values = labeled.apply(
            lambda row, f=forecaster: f.predict(row, history=history_lookup(row["timestamp"])).predicted_value,
            axis=1,
        )
        per_metric = pd.DataFrame({
            "timestamp": labeled["timestamp"].to_numpy(),
            f"forecast_{metric_name}": forecast_values.to_numpy(),
            f"current_{metric_name}": labeled["current_value"].to_numpy(),
            f"__target_{metric_name}": labeled["target"].to_numpy(),
        })
        combined = combined.merge(per_metric, on="timestamp", how="inner")

    if combined.empty:
        return combined

    threshold_columns = []
    for metric_name in deployed_forecasters:
        threshold = labeling_config.value_thresholds.get(metric_name)
        target_col = f"__target_{metric_name}"
        if threshold is None or target_col not in combined.columns:
            continue
        threshold_columns.append(combined[target_col] >= threshold)

    if threshold_columns:
        exceeds_any = threshold_columns[0]
        for col in threshold_columns[1:]:
            exceeds_any = exceeds_any | col
        combined["label"] = exceeds_any.astype(int)
    else:
        combined["label"] = 0

    combined = combined.drop(columns=[c for c in combined.columns if c.startswith("__target_")])
    return combined.reset_index(drop=True)


def build_live_contention_input_row(
    samples: Sequence[SystemMetricSample],
    feature_vector: FeatureVectorRecord,
    deployed_forecasters: dict[str, Forecaster],
    anomaly_detector: AnomalyDetector,
    feature_config: FeatureEngineeringConfig,
) -> pd.Series | None:
    """The live-inference counterpart to :func:`build_contention_dataset`.

    Builds exactly one ``forecast_{metric}``/``current_{metric}``/
    ``anomaly_score`` row for the *current* moment — i.e. what
    :func:`aiperfopt.contention.prediction.predict_contention` needs as
    its ``feature_row`` — without needing a forward-looking label
    (there is no "future" yet for a live cycle, unlike training/holdout
    evaluation). ``samples`` should be the same trailing window used to
    compute ``feature_vector`` (e.g. via
    :func:`aiperfopt.features.storage_bridge.compute_live_feature_vector`),
    so ``current_{metric}``/history lookups line up with the same moment
    the engineered features and forecasts describe.

    Also carries a ``confidence_{metric}`` column per target — not one of
    the classifier's model features (``ContentionClassifier.predict_proba``
    reindexes to its own trained ``feature_columns`` and ignores it), but
    the live-inference loop needs each forecaster's per-call confidence to
    persist alongside its prediction (Phase 7), and this is the one place
    that already calls ``forecaster.predict()`` per metric — recomputing
    it a second time downstream would double the cost of the classical
    (Holt/ETS) baseline's per-tick fit, which Phase 2's overhead budget
    exists precisely to keep in check.

    Returns ``None`` if there are no deployed forecasters, or if any
    target metric has no valid raw sample to read a current value from
    (mirrors ``build_contention_dataset``'s per-metric emptiness check,
    just for one row instead of a batch).
    """
    if not deployed_forecasters:
        return None

    feature_row = pd.Series(feature_vector.feature_values)
    anomaly_score = float(anomaly_detector.score(feature_row))

    result: dict[str, float | None] = {"timestamp": feature_vector.timestamp, "anomaly_score": anomaly_score}

    for metric_name, forecaster in deployed_forecasters.items():
        is_counter = metric_name in feature_config.counter_metrics

        raw_df = pd.DataFrame.from_records(
            [{"timestamp": s.timestamp, metric_name: getattr(s, metric_name, None)} for s in samples]
        )
        raw_df = raw_df.dropna(subset=[metric_name]).sort_values("timestamp")
        if raw_df.empty:
            return None

        raw_df["_dt"] = epoch_seconds_to_utc_datetime(raw_df["timestamp"])
        raw_series = raw_df.set_index("_dt")[metric_name].astype(float)
        if is_counter:
            raw_series = counter_rate_per_second(raw_series, max_gap_seconds=30.0)
        raw_series = raw_series.dropna()
        if raw_series.empty:
            return None

        current_value = float(raw_series.iloc[-1])
        history_lookup = build_history_lookup(raw_series)
        # Baseline forecasters (naive persistence, moving average) read
        # "current_value" directly off the row passed to .predict() —
        # mirroring build_supervised_dataset's per-metric row shape
        # (training path), which always carries that metric's own
        # current_value alongside the engineered features. The engineered
        # feature_row alone has no such column, so it's added per-metric
        # here rather than once globally, since each forecaster's
        # "current value" is a different metric's value.
        predict_row = feature_row.copy()
        predict_row["current_value"] = current_value
        forecast_result = forecaster.predict(
            predict_row, history=history_lookup(feature_vector.timestamp)
        )

        result[f"forecast_{metric_name}"] = forecast_result.predicted_value
        result[f"current_{metric_name}"] = current_value
        result[f"confidence_{metric_name}"] = forecast_result.confidence

    return pd.Series(result)
