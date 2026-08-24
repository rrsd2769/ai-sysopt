"""
Monitoring Engine (Section 7.1, item 1).

Implements the producer side of the explicit thread + queue
producer-consumer pattern required by ADL-18 (chosen over
``asyncio.run_in_executor`` specifically because a plain
``threading.Thread`` + ``queue.Queue`` is the clearer demonstration of
the underlying OS concept for this course project).

Two independent polling threads run concurrently:

- **System metrics thread** — polls at ``monitoring.polling_interval_seconds``
  (NFR-2.1 default 1-2s), producing one :class:`SystemMetricSample` per
  cycle.
- **Process metrics thread** — polls at
  ``monitoring.process_polling_interval_seconds`` (NFR-2.2: coarser than
  aggregate metrics is explicitly allowed), producing
  :class:`ProcessMetricSample` rows plus any launch/termination
  :class:`ProcessEvent` rows detected that cycle (FR-1.3).

Both threads push onto one shared, bounded ``queue.Queue`` (FR-1.2: "on a
dedicated thread independent of the API event loop" — there is no event
loop yet in Phase 2, but the separation from the future API loop is
already structurally guaranteed by these being plain OS threads).

Backpressure handling
----------------------
FR-2.2 requires Ingestion to never block Monitoring regardless of storage
write latency. If Ingestion falls behind and the queue fills up, a
*blocking* ``queue.put()`` here would defeat that guarantee by blocking
the producer instead. Producers therefore use ``put_nowait`` and, on
``queue.Full``, drop the sample and increment ``dropped_sample_count``
rather than blocking — a full queue means the system is already
overloaded, and Monitoring's job (keep sampling) takes priority over any
single sample's delivery.
"""

from __future__ import annotations

import queue
import threading
import time

from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import MonitoringError
from aiperfopt.config.settings import MonitoringSettings
from aiperfopt.monitoring.collectors import collect_system_metrics, prime_cpu_percent_counters
from aiperfopt.monitoring.process_tracker import ProcessTracker
from aiperfopt.monitoring.queue_items import QueueItem

logger = get_logger(__name__)


class MonitoringEngine:
    """Owns the two polling threads and the shared output queue.

    Parameters
    ----------
    settings:
        The monitoring section of :class:`~aiperfopt.config.settings.AppSettings`.
    output_queue:
        The shared queue this engine produces onto. Constructed by the
        caller (see :mod:`aiperfopt.pipeline`) so Monitoring and Ingestion
        share the same queue instance without either owning the other's
        lifecycle.
    excluded_process_names:
        Process names to exclude from collection entirely (Section 13,
        NFR-7.4). Matched case-insensitively.
    """

    def __init__(
        self,
        settings: MonitoringSettings,
        output_queue: "queue.Queue[QueueItem]",
        excluded_process_names: frozenset[str] = frozenset(),
    ) -> None:
        self._settings = settings
        self._queue = output_queue
        self._process_tracker = ProcessTracker(excluded_process_names)

        self._stop_event = threading.Event()
        self._system_thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None

        self._dropped_sample_count = 0
        self._lock = threading.Lock()

    @property
    def dropped_sample_count(self) -> int:
        with self._lock:
            return self._dropped_sample_count

    @property
    def is_running(self) -> bool:
        return bool(
            self._system_thread and self._system_thread.is_alive()
            and self._process_thread and self._process_thread.is_alive()
        )

    def start(self) -> None:
        if self.is_running:
            return

        prime_cpu_percent_counters()
        self._stop_event.clear()

        self._system_thread = threading.Thread(
            target=self._run_system_metrics_loop, name="aiperfopt-monitor-system", daemon=True
        )
        self._process_thread = threading.Thread(
            target=self._run_process_metrics_loop, name="aiperfopt-monitor-process", daemon=True
        )
        self._system_thread.start()
        self._process_thread.start()
        logger.info(
            "Monitoring Engine started",
            extra={
                "polling_interval_seconds": self._settings.polling_interval_seconds,
                "process_polling_interval_seconds": self._settings.process_polling_interval_seconds,
            },
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal both polling threads to stop and wait for them to exit.

        A ``timeout`` bounds how long shutdown can take; if a thread
        doesn't exit in time this is logged (it indicates a poll cycle is
        stuck) rather than hanging the caller forever.
        """
        self._stop_event.set()
        for thread in (self._system_thread, self._process_thread):
            if thread is not None:
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning("Monitoring thread did not stop within timeout", extra={"thread": thread.name})
        logger.info("Monitoring Engine stopped", extra={"dropped_sample_count": self.dropped_sample_count})

    def _put(self, item: QueueItem) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self._dropped_sample_count += 1
            logger.warning(
                "Monitoring->Ingestion queue full, dropping sample",
                extra={"kind": item.kind.value, "dropped_sample_count": self._dropped_sample_count},
            )

    def _run_system_metrics_loop(self) -> None:
        interval = self._settings.polling_interval_seconds
        while not self._stop_event.is_set():
            cycle_start = time.monotonic()
            try:
                sample = collect_system_metrics()
                self._put(QueueItem.system_metric(sample))
            except Exception as exc:  # noqa: BLE001 - deliberate: see FR-1.4/NFR-5.1
                # collect_system_metrics() only lets exceptions through
                # for its two mandatory sources; a failure here means the
                # environment itself is broken for this cycle. Per
                # NFR-5.1, this must not crash the engine — log and retry
                # next cycle.
                logger.error(
                    "System metrics collection cycle failed",
                    exc_info=True,
                    extra={"error": str(exc)},
                )
            self._sleep_remaining(cycle_start, interval)

    def _run_process_metrics_loop(self) -> None:
        interval = self._settings.process_polling_interval_seconds
        while not self._stop_event.is_set():
            cycle_start = time.monotonic()
            try:
                result = self._process_tracker.poll()
                for sample in result.metrics:
                    self._put(QueueItem.process_metric(sample))
                for event in result.events:
                    self._put(QueueItem.process_event(event))
            except Exception as exc:  # noqa: BLE001 - see rationale above
                logger.error(
                    "Process metrics collection cycle failed",
                    exc_info=True,
                    extra={"error": str(exc)},
                )
            self._sleep_remaining(cycle_start, interval)

    def _sleep_remaining(self, cycle_start: float, interval: float) -> None:
        elapsed = time.monotonic() - cycle_start
        remaining = interval - elapsed
        if remaining > 0:
            # Sleep in short slices so `stop()` is responsive even during
            # a long configured interval, rather than one uninterruptible
            # time.sleep(interval) call.
            deadline = time.monotonic() + remaining
            while not self._stop_event.is_set() and time.monotonic() < deadline:
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        elif elapsed > interval * 2:
            logger.warning(
                "Monitoring poll cycle exceeded its interval",
                extra={"elapsed_seconds": elapsed, "interval_seconds": interval},
            )
