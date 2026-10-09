"""
Wires the Monitoring Engine and Data Ingestion consumer together around
their shared queue.

:func:`aiperfopt.bootstrap.bootstrap` deliberately does *not* start any
background threads itself — it only prepares config/logging/storage.
Starting the live pipeline is a distinct, explicit action, taken here by
whichever entrypoint wants a running monitor (a future CLI command, the
API server's startup hook, or a test/demo script). Keeping this
separate from ``bootstrap()`` means importing the package, or running its
test suite, never has the side effect of spawning threads.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass

from aiperfopt.applog import get_logger
from aiperfopt.bootstrap import AppContext
from aiperfopt.monitoring.engine import MonitoringEngine
from aiperfopt.monitoring.ingestion import IngestionConsumer
from aiperfopt.monitoring.queue_items import QueueItem

logger = get_logger(__name__)


@dataclass
class MonitoringPipeline:
    """Bundles the Monitoring Engine, Ingestion Consumer, and their shared queue."""

    engine: MonitoringEngine
    consumer: IngestionConsumer
    queue: "queue.Queue[QueueItem]"

    def start(self) -> None:
        # Ingestion starts first so it's ready to drain the queue the
        # instant Monitoring's first sample lands.
        self.consumer.start()
        self.engine.start()

    def stop(self, timeout: float = 5.0) -> None:
        # Monitoring stops first so no new samples arrive mid-shutdown;
        # Ingestion then drains and flushes whatever's left in the queue
        # (see IngestionConsumer._run's should_stop + queue.empty() exit
        # condition) before its own thread exits.
        self.engine.stop(timeout=timeout)
        self.consumer.stop(timeout=timeout)


def create_monitoring_pipeline(ctx: AppContext) -> MonitoringPipeline:
    """Build a ready-to-start :class:`MonitoringPipeline` from a bootstrap context."""
    settings = ctx.settings
    shared_queue: "queue.Queue[QueueItem]" = queue.Queue(maxsize=settings.ingestion.queue_maxsize)

    excluded = frozenset(name.lower() for name in settings.privacy.excluded_processes)

    engine = MonitoringEngine(
        settings=settings.monitoring,
        output_queue=shared_queue,
        excluded_process_names=excluded,
    )
    consumer = IngestionConsumer(
        storage_engine=ctx.storage_engine,
        input_queue=shared_queue,
        settings=settings.ingestion,
        full_resolution_retention_hours=settings.storage.full_resolution_retention_hours,
        process_retention_hours=settings.storage.process_retention_hours,
    )

    logger.info("Monitoring pipeline assembled")
    return MonitoringPipeline(engine=engine, consumer=consumer, queue=shared_queue)
