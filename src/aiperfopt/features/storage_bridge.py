"""
Storage-backed Feature Engineering entry points.

:mod:`aiperfopt.features.engine` has no dependency on storage at all —
it's pure functions over lists of :class:`SystemMetricSample`, which is
what makes it trivially unit-testable and guarantees FR-4.2 (the same
function serves both paths). This module is the thin, storage-aware layer
on top of it that Forecasting (Phase 4) and any live-inference loop will
actually call: read the right window from storage, run the shared
computation, optionally persist the result.
"""

from __future__ import annotations

import time

import pandas as pd
from sqlalchemy import Engine

from aiperfopt.applog import get_logger
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.features.engine import (
    compute_feature_matrix,
    compute_latest_feature_vector,
    feature_matrix_to_records,
)
from aiperfopt.features.process_features import PROCESS_SUMMARY_COLUMNS, compute_process_summary_series
from aiperfopt.storage.reader import (
    get_process_events_range,
    get_process_metric_aggregates_range,
    get_process_metrics_range,
    get_recent_system_metrics,
    get_system_metrics_range,
)
from aiperfopt.storage.records import FeatureVectorRecord, ProcessFeatureSummaryRecord
from aiperfopt.storage.writer import (
    insert_feature_vector,
    insert_feature_vectors_batch,
    insert_process_feature_summaries_batch,
)

logger = get_logger(__name__)


def compute_live_feature_vector(
    engine: Engine, config: FeatureEngineeringConfig, as_of: float | None = None
) -> FeatureVectorRecord | None:
    """Read the trailing window needed for ``config`` and compute one feature vector.

    This is Section 7.3's "historical window read on every cycle" for
    live inference: Feature Engineering depends on Data Storage even at
    inference time, not just for training. The window size is derived
    from ``config.min_history_seconds`` so every configured lag/rolling
    window has enough data to be computed rather than coming back
    ``None`` purely for lack of history.
    """
    samples = get_recent_system_metrics(engine, window_seconds=config.min_history_seconds, as_of=as_of)
    return compute_latest_feature_vector(samples, config)


def compute_and_store_live_feature_vector(
    engine: Engine, config: FeatureEngineeringConfig, as_of: float | None = None
) -> FeatureVectorRecord | None:
    """Compute the live feature vector and persist it to the ``features`` table."""
    record = compute_live_feature_vector(engine, config, as_of=as_of)
    if record is None:
        logger.debug("No feature vector computed — insufficient trailing data")
        return None
    record.id = insert_feature_vector(engine, record)
    return record


def backfill_feature_matrix(
    engine: Engine,
    config: FeatureEngineeringConfig,
    start: float,
    end: float,
    persist: bool = True,
) -> int:
    """Compute features over a historical range and optionally persist all rows.

    Intended for building a training dataset ahead of Forecasting model
    training (Phase 4): every row of the resulting matrix is a training
    example, computed by the exact same formulas the live path uses
    (FR-4.2). Returns the number of feature vectors computed.
    """
    samples = get_system_metrics_range(engine, start=start, end=end)
    matrix = compute_feature_matrix(samples, config)
    if matrix.empty:
        logger.info("Backfill produced no feature rows", extra={"start": start, "end": end})
        return 0

    records = feature_matrix_to_records(matrix)
    if persist:
        insert_feature_vectors_batch(engine, records)

    logger.info(
        "Feature backfill complete",
        extra={"rows": len(records), "start": start, "end": end, "persisted": persist},
    )
    return len(records)


def _records_to_frame(records: list, columns: tuple[str, ...]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=list(columns))
    return pd.DataFrame([{c: getattr(r, c) for c in columns} for r in records])


def summarize_and_persist_process_features(engine: Engine, cutoff: float) -> int:
    """Compute and persist the process summary series for rows about to be pruned.

    Must be called with the same ``cutoff`` immediately *before*
    :func:`aiperfopt.storage.rollup.prune_process_level_data` (see
    ``aiperfopt.ingestion.consumer``'s retention check, which does exactly
    this) — the raw ``raw_process_metrics``/``process_events``/
    ``raw_process_metrics_other`` rows this reads are exactly the ones
    about to be deleted.

    Because both this function and the prune that follows it always
    operate on "everything older than the current cutoff", and the raw
    rows are deleted immediately after being summarized, a given raw
    timestamp is summarized exactly once across the process's lifetime —
    no watermark bookkeeping needed.

    Returns the number of summary rows written.
    """
    # `get_*_range`'s `end` bound is inclusive (<=); prune_process_level_data's
    # cutoff is exclusive (<). Filtering to `timestamp < cutoff` below keeps
    # the summarized row-set identical to the row-set about to be deleted,
    # so a boundary-timestamp row is never summarized twice across
    # consecutive retention checks.
    metrics = [m for m in get_process_metrics_range(engine, end=cutoff) if m.timestamp < cutoff]
    if not metrics:
        return 0
    events = [e for e in get_process_events_range(engine, end=cutoff) if e.timestamp < cutoff]
    aggregates = [
        a for a in get_process_metric_aggregates_range(engine, end=cutoff) if a.timestamp < cutoff
    ]

    metrics_df = _records_to_frame(
        metrics, ("timestamp", "pid", "cpu_percent", "memory_rss", "num_threads", "create_time")
    )
    events_df = _records_to_frame(events, ("timestamp", "event_type"))
    aggregates_df = _records_to_frame(
        aggregates, ("timestamp", "other_count", "other_cpu_sum", "other_mem_rss_sum")
    )

    summary_df = compute_process_summary_series(metrics_df, events_df, aggregates_df)
    if summary_df.empty:
        return 0

    computed_at = time.time()
    records_out = [
        ProcessFeatureSummaryRecord(
            timestamp=float(row["timestamp"]),
            computed_at=computed_at,
            summary_values={col: float(row[col]) for col in PROCESS_SUMMARY_COLUMNS},
        )
        for _, row in summary_df.iterrows()
    ]
    insert_process_feature_summaries_batch(engine, records_out)
    logger.info(
        "Process summary series persisted before prune",
        extra={"rows": len(records_out), "cutoff": cutoff},
    )
    return len(records_out)
