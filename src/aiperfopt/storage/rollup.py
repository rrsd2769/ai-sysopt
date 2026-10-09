"""
Downsampling / retention (FR-3.2, NFR-4.1, NFR-4.3).

Two distinct strategies are applied to old data, by design:

- **System metrics are rolled up**, not deleted outright: raw rows older
  than the retention cutoff are aggregated into hourly (by default)
  buckets in ``raw_system_metrics_rollup`` and then removed, in a single
  transaction. System-level trend history has ongoing value for Reporting
  (FR-14.1) and for understanding long-run workload patterns, so it is
  worth preserving in a bounded, downsampled form rather than discarding
  it.
- **Process-level rows and process events are pruned outright**, not
  rolled up. Per-process history is high-cardinality (unique per pid,
  churns constantly as processes start/stop) and a meaningful rollup
  would require tracking per-process identity across buckets — real
  complexity for data whose value is concentrated in the recent window
  used for diagnosis, not in long-run per-process trends. This is a
  deliberate scope decision, not an oversight: NFR-4.3 requires storage
  growth to remain "predictable and bounded", which pruning satisfies
  without the added complexity.

Neither this module's functions are invoked on any schedule in Phase 1 —
that hookup (a periodic maintenance task) belongs to whichever subsystem
owns the running event loop, starting with Monitoring/Ingestion in
Phase 2. Phase 1 provides the mechanism; Phase 2+ provides the trigger.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import Engine, delete, func, select
from sqlalchemy.exc import SQLAlchemyError

from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import StorageError
from aiperfopt.storage.schema import (
    process_events,
    raw_process_metrics,
    raw_process_metrics_other,
    raw_system_metrics,
    raw_system_metrics_rollup,
)

logger = get_logger(__name__)

DEFAULT_BUCKET_SECONDS = 3600  # 1 hour

# Gauge-style metrics get avg/min/max in the rollup; counter-style
# (monotonically-accumulating OS counters) get sum. This list is the one
# place that needs updating if Section 9's metric set changes — the
# rollup table itself (schema.py) never needs a migration for it, since
# aggregates are stored as JSON.
_GAUGE_METRICS = ("cpu_percent", "mem_percent", "load_avg_1", "load_avg_5", "load_avg_15",
                   "battery_percent")
_COUNTER_METRICS = ("disk_read_bytes", "disk_write_bytes", "net_bytes_sent", "net_bytes_recv")


@dataclass(frozen=True)
class RollupResult:
    buckets_written: int
    raw_rows_removed: int


@dataclass(frozen=True)
class PruneResult:
    process_metric_rows_removed: int
    process_event_rows_removed: int
    process_aggregate_rows_removed: int = 0


def rollup_system_metrics(
    engine: Engine, cutoff: float, bucket_seconds: int = DEFAULT_BUCKET_SECONDS
) -> RollupResult:
    """Aggregate ``raw_system_metrics`` rows older than ``cutoff`` into buckets.

    Aggregation and the subsequent delete of source rows happen in a
    single transaction, so a crash mid-rollup cannot leave data
    double-counted (present in both the raw table and a rollup bucket) or
    silently dropped (removed from raw without a corresponding bucket).

    Parameters
    ----------
    cutoff:
        Unix epoch seconds. Only rows with ``timestamp < cutoff`` are
        rolled up; more recent rows are left at full resolution.
    bucket_seconds:
        Width of each aggregation bucket. Defaults to one hour.
    """
    try:
        with engine.begin() as connection:
            rows = connection.execute(
                select(raw_system_metrics).where(raw_system_metrics.c.timestamp < cutoff)
            ).fetchall()

            if not rows:
                return RollupResult(buckets_written=0, raw_rows_removed=0)

            buckets: dict[float, list] = defaultdict(list)
            for row in rows:
                bucket_start = (row.timestamp // bucket_seconds) * bucket_seconds
                buckets[bucket_start].append(row)

            bucket_payloads = []
            for bucket_start, bucket_rows in buckets.items():
                aggregates: dict[str, dict[str, float]] = {}
                for metric in _GAUGE_METRICS:
                    values = [getattr(r, metric) for r in bucket_rows if getattr(r, metric) is not None]
                    if values:
                        aggregates[metric] = {
                            "avg": sum(values) / len(values),
                            "min": min(values),
                            "max": max(values),
                        }
                for metric in _COUNTER_METRICS:
                    values = [getattr(r, metric) for r in bucket_rows if getattr(r, metric) is not None]
                    if values:
                        aggregates[metric] = {"sum": sum(values)}

                bucket_payloads.append(
                    {
                        "bucket_start": bucket_start,
                        "bucket_seconds": bucket_seconds,
                        "sample_count": len(bucket_rows),
                        "aggregates": aggregates,
                    }
                )

            connection.execute(raw_system_metrics_rollup.insert(), bucket_payloads)

            ids_to_remove = [row.id for row in rows]
            connection.execute(
                delete(raw_system_metrics).where(raw_system_metrics.c.id.in_(ids_to_remove))
            )

        logger.info(
            "System metrics rolled up",
            extra={
                "buckets_written": len(bucket_payloads),
                "raw_rows_removed": len(rows),
                "cutoff": cutoff,
            },
        )
        return RollupResult(buckets_written=len(bucket_payloads), raw_rows_removed=len(rows))

    except SQLAlchemyError as exc:
        raise StorageError("Rollup of raw_system_metrics failed", context={"cutoff": cutoff}) from exc


def prune_process_level_data(engine: Engine, cutoff: float) -> PruneResult:
    """Delete process-level rows, process events and aggregate rows older than ``cutoff``.

    See the module docstring for why this is a straight prune rather than
    a rollup. The "everything else" aggregate rows (``raw_process_metrics_
    other``) are pruned on the same cutoff as the per-process rows they
    were computed alongside -- they exist to make the top-N-bounded
    per-process rows sum-correct (Phase 2), which is only meaningful while
    those per-process rows are still present.
    """
    try:
        with engine.begin() as connection:
            metrics_result = connection.execute(
                delete(raw_process_metrics).where(raw_process_metrics.c.timestamp < cutoff)
            )
            events_result = connection.execute(
                delete(process_events).where(process_events.c.timestamp < cutoff)
            )
            aggregates_result = connection.execute(
                delete(raw_process_metrics_other).where(raw_process_metrics_other.c.timestamp < cutoff)
            )

        result = PruneResult(
            process_metric_rows_removed=metrics_result.rowcount or 0,
            process_event_rows_removed=events_result.rowcount or 0,
            process_aggregate_rows_removed=aggregates_result.rowcount or 0,
        )
        logger.info("Process-level data pruned", extra={"cutoff": cutoff, **result.__dict__})
        return result
    except SQLAlchemyError as exc:
        raise StorageError("Prune of process-level data failed", context={"cutoff": cutoff}) from exc


def apply_retention_policy(
    engine: Engine,
    full_resolution_retention_hours: float,
    process_retention_hours: float,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
    as_of: float | None = None,
) -> tuple[RollupResult, PruneResult]:
    """Apply the full retention policy: roll up old system metrics, prune old process data.

    This is the single function a scheduler (the ingestion consumer, see
    ``aiperfopt.monitoring.ingestion``) calls periodically. The two
    retention windows are independent by design (Phase 2): process-level
    detail needs to be pruned much sooner than system metrics to keep
    multi-day passive collection bounded (NFR-4.3), while system-level
    trend history is rolled up rather than deleted and can afford to stay
    at full resolution far longer.  ``full_resolution_retention_hours``
    and ``process_retention_hours`` are expected to come from
    ``AppSettings.storage`` (NFR-4.1).
    """
    reference = as_of if as_of is not None else time.time()
    system_cutoff = reference - (full_resolution_retention_hours * 3600)
    process_cutoff = reference - (process_retention_hours * 3600)

    rollup_result = rollup_system_metrics(engine, system_cutoff, bucket_seconds)
    prune_result = prune_process_level_data(engine, process_cutoff)
    return rollup_result, prune_result
