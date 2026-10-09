"""
The item type carried on the shared Monitoring -> Ingestion queue.

Monitoring Engine threads are the only producers; the Data Ingestion
consumer thread is the only consumer (the explicit producer-consumer
pattern). Wrapping every payload in a tagged ``QueueItem`` lets a single
shared ``queue.Queue`` carry all three raw-sample record types — system
metrics, process metrics, and process events — without the consumer
needing three separate queues (which would complicate the ordering and
backpressure story for no real benefit, since a single
``threading.Thread`` consumer processes one queue in strict FIFO order
regardless).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from aiperfopt.storage.records import (
    ProcessEvent,
    ProcessMetricAggregate,
    ProcessMetricSample,
    SystemMetricSample,
)


class QueueItemKind(str, Enum):
    SYSTEM_METRIC = "system_metric"
    PROCESS_METRIC = "process_metric"
    PROCESS_EVENT = "process_event"
    PROCESS_METRIC_AGGREGATE = "process_metric_aggregate"


@dataclass(slots=True, frozen=True)
class QueueItem:
    kind: QueueItemKind
    payload: SystemMetricSample | ProcessMetricSample | ProcessEvent | ProcessMetricAggregate

    @staticmethod
    def system_metric(sample: SystemMetricSample) -> "QueueItem":
        return QueueItem(kind=QueueItemKind.SYSTEM_METRIC, payload=sample)

    @staticmethod
    def process_metric(sample: ProcessMetricSample) -> "QueueItem":
        return QueueItem(kind=QueueItemKind.PROCESS_METRIC, payload=sample)

    @staticmethod
    def process_event(event: ProcessEvent) -> "QueueItem":
        return QueueItem(kind=QueueItemKind.PROCESS_EVENT, payload=event)

    @staticmethod
    def process_metric_aggregate(aggregate: ProcessMetricAggregate) -> "QueueItem":
        return QueueItem(kind=QueueItemKind.PROCESS_METRIC_AGGREGATE, payload=aggregate)
