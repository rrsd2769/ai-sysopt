"""
Data Ingestion / Collector (Section 7.1, item 2).

The consumer side of ADL-18's producer-consumer pattern: one dedicated
thread drains the shared queue populated by
:class:`~aiperfopt.monitoring.engine.MonitoringEngine`, accumulates
same-type records into batches, and flushes each batch to storage via
:mod:`aiperfopt.storage.writer`'s batch insert functions (FR-2.1).

Ordering (FR-2.3)
------------------
``queue.Queue`` is strictly FIFO, and this module uses exactly one
consumer thread, so items are dequeued in the exact order Monitoring
produced them. Each record type accumulates into its own list in dequeue
order, so a flushed batch is written to storage in the same chronological
order it was collected. (Storage's reader layer also always applies an
explicit ``ORDER BY`` — see :mod:`aiperfopt.storage.reader` — so
chronological correctness never actually depends on physical insertion
order; FIFO consumption here is what makes the *ordering guarantee*
provable rather than incidental.)

Non-blocking with respect to Monitoring (FR-2.2)
--------------------------------------------------
This consumer runs on its own thread entirely separate from Monitoring's
two producer threads. A slow storage write here can only ever make this
thread's own batches accumulate for longer — it cannot block a
``MonitoringEngine`` poll cycle, since the two communicate exclusively
through the (non-blocking-to-write, per ``engine.py``) shared queue.

Failure isolation (NFR-5.4)
------------------------------
A storage failure during flush is logged and the batch is dropped rather
than retried in a tight loop or allowed to crash the consumer thread —
retrying indefinitely against a persistently broken storage layer would
itself become an unbounded-memory backpressure problem, and per NFR-5.4 a
failure in this subsystem must not halt Monitoring, which continues
regardless of Ingestion's state.
"""

from __future__ import annotations

import queue
import threading
import time

from sqlalchemy import Engine

from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import StorageError
from aiperfopt.config.settings import IngestionSettings
from aiperfopt.features.storage_bridge import summarize_and_persist_process_features
from aiperfopt.monitoring.queue_items import QueueItem, QueueItemKind
from aiperfopt.storage.records import (
    ProcessEvent,
    ProcessMetricAggregate,
    ProcessMetricSample,
    SystemMetricSample,
)
from aiperfopt.storage.rollup import apply_retention_policy
from aiperfopt.storage.writer import (
    insert_process_events_batch,
    insert_process_metric_aggregates_batch,
    insert_process_metric_samples_batch,
    insert_system_metric_samples_batch,
)

logger = get_logger(__name__)

# How long a single queue.get() call blocks before re-checking the stop
# flag / flush-interval deadline. Small enough that stop() is responsive,
# large enough to avoid busy-waiting.
_QUEUE_POLL_TIMEOUT_SECONDS = 0.1


class IngestionConsumer:
    """Drains the shared queue and batches writes into storage.

    Parameters
    ----------
    storage_engine:
        The engine batches are written through.
    input_queue:
        The queue shared with a :class:`~aiperfopt.monitoring.engine.MonitoringEngine`.
    settings:
        Ingestion tunables (batch size, flush interval, retention check
        interval) — see :class:`~aiperfopt.config.settings.IngestionSettings`.
    full_resolution_retention_hours, process_retention_hours:
        Passed through to :func:`aiperfopt.storage.rollup.apply_retention_policy`
        on each retention check — independent windows, see that
        function's docstring for why (Phase 2, NFR-4.3).
    """

    def __init__(
        self,
        storage_engine: Engine,
        input_queue: "queue.Queue[QueueItem]",
        settings: IngestionSettings,
        full_resolution_retention_hours: float,
        process_retention_hours: float,
    ) -> None:
        self._engine = storage_engine
        self._queue = input_queue
        self._settings = settings
        self._retention_hours = full_resolution_retention_hours
        self._process_retention_hours = process_retention_hours

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._flushed_counts = {kind: 0 for kind in QueueItemKind}
        self._dropped_on_flush_failure = {kind: 0 for kind in QueueItemKind}
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def flushed_count(self, kind: QueueItemKind) -> int:
        with self._lock:
            return self._flushed_counts[kind]

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="aiperfopt-ingestion-consumer", daemon=True
        )
        self._thread.start()
        logger.info(
            "Ingestion consumer started",
            extra={
                "batch_size": self._settings.batch_size,
                "flush_interval_seconds": self._settings.flush_interval_seconds,
            },
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Ingestion consumer did not stop within timeout")
        logger.info("Ingestion consumer stopped", extra={"flushed_counts": self._flushed_counts})

    def _run(self) -> None:
        system_batch: list[SystemMetricSample] = []
        process_batch: list[ProcessMetricSample] = []
        event_batch: list[ProcessEvent] = []
        aggregate_batch: list[ProcessMetricAggregate] = []

        last_flush = time.monotonic()
        last_retention_check = time.monotonic()

        while True:
            try:
                item = self._queue.get(timeout=_QUEUE_POLL_TIMEOUT_SECONDS)
                self._route(item, system_batch, process_batch, event_batch, aggregate_batch)
            except queue.Empty:
                pass

            batch_full = (
                len(system_batch) >= self._settings.batch_size
                or len(process_batch) >= self._settings.batch_size
                or len(event_batch) >= self._settings.batch_size
                or len(aggregate_batch) >= self._settings.batch_size
            )
            interval_elapsed = (
                time.monotonic() - last_flush >= self._settings.flush_interval_seconds
            )
            should_stop = self._stop_event.is_set()

            if batch_full or interval_elapsed or should_stop:
                if system_batch or process_batch or event_batch or aggregate_batch:
                    self._flush(system_batch, process_batch, event_batch, aggregate_batch)
                    system_batch, process_batch, event_batch, aggregate_batch = [], [], [], []
                last_flush = time.monotonic()

            if (
                time.monotonic() - last_retention_check
                >= self._settings.retention_check_interval_seconds
            ):
                self._run_retention_check()
                last_retention_check = time.monotonic()

            if should_stop and self._queue.empty():
                break

    def _route(
        self,
        item: QueueItem,
        system_batch: list[SystemMetricSample],
        process_batch: list[ProcessMetricSample],
        event_batch: list[ProcessEvent],
        aggregate_batch: list[ProcessMetricAggregate],
    ) -> None:
        if item.kind is QueueItemKind.SYSTEM_METRIC:
            system_batch.append(item.payload)  # type: ignore[arg-type]
        elif item.kind is QueueItemKind.PROCESS_METRIC:
            process_batch.append(item.payload)  # type: ignore[arg-type]
        elif item.kind is QueueItemKind.PROCESS_EVENT:
            event_batch.append(item.payload)  # type: ignore[arg-type]
        elif item.kind is QueueItemKind.PROCESS_METRIC_AGGREGATE:
            aggregate_batch.append(item.payload)  # type: ignore[arg-type]

    def _flush(
        self,
        system_batch: list[SystemMetricSample],
        process_batch: list[ProcessMetricSample],
        event_batch: list[ProcessEvent],
        aggregate_batch: list[ProcessMetricAggregate],
    ) -> None:
        self._flush_one(QueueItemKind.SYSTEM_METRIC, system_batch, insert_system_metric_samples_batch)
        self._flush_one(QueueItemKind.PROCESS_METRIC, process_batch, insert_process_metric_samples_batch)
        self._flush_one(QueueItemKind.PROCESS_EVENT, event_batch, insert_process_events_batch)
        self._flush_one(
            QueueItemKind.PROCESS_METRIC_AGGREGATE, aggregate_batch, insert_process_metric_aggregates_batch
        )

    def _flush_one(self, kind: QueueItemKind, batch: list, writer_fn) -> None:  # type: ignore[no-untyped-def]
        if not batch:
            return
        try:
            writer_fn(self._engine, batch)
            with self._lock:
                self._flushed_counts[kind] += len(batch)
        except StorageError:
            with self._lock:
                self._dropped_on_flush_failure[kind] += len(batch)
            logger.error(
                "Batch flush to storage failed; batch dropped",
                exc_info=True,
                extra={"kind": kind.value, "batch_size": len(batch)},
            )

    def _run_retention_check(self) -> None:
        try:
            # Summarize process-level data before pruning it (Phase 2):
            # the derived process_feature_summaries rows this writes must
            # exist before prune_process_level_data (inside
            # apply_retention_policy) deletes the raw rows they're
            # computed from. Both use the same `as_of` reference so the
            # two operate on identical cutoffs.
            as_of = time.time()
            process_cutoff = as_of - (self._process_retention_hours * 3600)
            summarized = summarize_and_persist_process_features(self._engine, process_cutoff)

            rollup_result, prune_result = apply_retention_policy(
                self._engine, self._retention_hours, self._process_retention_hours, as_of=as_of
            )
            logger.info(
                "Retention policy applied",
                extra={
                    "buckets_written": rollup_result.buckets_written,
                    "raw_rows_removed": rollup_result.raw_rows_removed,
                    "process_metric_rows_removed": prune_result.process_metric_rows_removed,
                    "process_event_rows_removed": prune_result.process_event_rows_removed,
                    "process_aggregate_rows_removed": prune_result.process_aggregate_rows_removed,
                    "process_summary_rows_written": summarized,
                },
            )
        except StorageError:
            logger.error("Retention policy check failed", exc_info=True)
