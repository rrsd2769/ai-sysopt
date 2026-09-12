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
from aiperfopt.features.process_features import (
    PROCESS_SUMMARY_COLUMNS,
    align_process_summary_to_timestamps,
    compute_process_summary_series,
)
from aiperfopt.storage.reader import (
    get_process_events_range,
    get_process_feature_summaries_range,
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
    if not samples:
        return None
    process_summary = aligned_process_summary(engine, samples, config.tolerance_seconds)
    return compute_latest_feature_vector(samples, config, process_summary=process_summary)


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
    process_summary = aligned_process_summary(engine, samples, config.tolerance_seconds)
    matrix = compute_feature_matrix(samples, config, process_summary=process_summary)
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


def read_process_summary_series(engine: Engine, start: float, end: float) -> pd.DataFrame:
    """Per-poll-timestamp process summary covering ``[start, end]``, from whichever source has it.

    Phase 2's independent process retention window means a long
    historical range can straddle two different sources: rows older than
    the process-retention cutoff have already been summarized and their
    raw rows pruned (only ``process_feature_summaries`` survives for
    them — see :func:`summarize_and_persist_process_features`), while
    rows newer than the cutoff are still present as raw
    ``raw_process_metrics``/``process_events``/``raw_process_metrics_other``
    and haven't been summarized yet. This reads both and combines them
    into one frame shaped like
    :func:`aiperfopt.features.process_features.compute_process_summary_series`'s
    output — the only shape
    :func:`aiperfopt.features.process_features.align_process_summary_to_timestamps`
    needs, regardless of which source a given poll came from.
    """
    persisted = get_process_feature_summaries_range(engine, start=start, end=end)
    persisted_df = pd.DataFrame(
        [
            {"timestamp": r.timestamp, **{c: r.summary_values.get(c) for c in PROCESS_SUMMARY_COLUMNS}}
            for r in persisted
        ]
    ) if persisted else pd.DataFrame(columns=["timestamp", *PROCESS_SUMMARY_COLUMNS])

    metrics = get_process_metrics_range(engine, start=start, end=end)
    raw_df = pd.DataFrame(columns=["timestamp", *PROCESS_SUMMARY_COLUMNS])
    if metrics:
        metrics_df = _records_to_frame(
            metrics, ("timestamp", "pid", "cpu_percent", "memory_rss", "num_threads", "create_time")
        )
        events_df = _records_to_frame(
            get_process_events_range(engine, start=start, end=end), ("timestamp", "event_type")
        )
        aggregates_df = _records_to_frame(
            get_process_metric_aggregates_range(engine, start=start, end=end),
            ("timestamp", "other_count", "other_cpu_sum", "other_mem_rss_sum"),
        )
        raw_df = compute_process_summary_series(metrics_df, events_df, aggregates_df)

    non_empty = [df for df in (persisted_df, raw_df) if not df.empty]
    if not non_empty:
        return pd.DataFrame(columns=["timestamp", *PROCESS_SUMMARY_COLUMNS])
    combined = pd.concat(non_empty, ignore_index=True)
    # The two sources shouldn't overlap in timestamp by construction
    # (Phase 2 always summarizes a poll immediately before pruning its raw
    # rows) — keep() the raw-computed value on the rare chance both exist
    # for the same poll, since it's the more direct computation.
    return (
        combined.drop_duplicates(subset="timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def aligned_process_summary(
    engine: Engine, samples: list, tolerance_seconds: float
) -> pd.DataFrame | None:
    """Process summary aligned onto ``samples``' own timestamps, or ``None`` if there's none.

    Shared by the live (:func:`compute_live_feature_vector`), batch
    (:func:`backfill_feature_matrix`), and Phase 5 ablation
    (:mod:`aiperfopt.evaluation.ablation`) paths so all three feed
    :func:`aiperfopt.features.engine.compute_feature_matrix` through the
    identical alignment call — the guarantee that process columns come
    out identical between training and live for the same window. Public
    (not underscore-prefixed) since the ablation module is a legitimate
    caller from outside this module.
    """
    if not samples:
        return None
    timestamps = [s.timestamp for s in samples]
    summary = read_process_summary_series(engine, start=min(timestamps), end=max(timestamps))
    if summary.empty:
        return None
    return align_process_summary_to_timestamps(summary, timestamps, tolerance_seconds)


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
