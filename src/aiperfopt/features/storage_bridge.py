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

from sqlalchemy import Engine

from aiperfopt.applog import get_logger
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.features.engine import (
    compute_feature_matrix,
    compute_latest_feature_vector,
    feature_matrix_to_records,
)
from aiperfopt.storage.reader import get_recent_system_metrics, get_system_metrics_range
from aiperfopt.storage.records import FeatureVectorRecord
from aiperfopt.storage.writer import insert_feature_vector, insert_feature_vectors_batch

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
