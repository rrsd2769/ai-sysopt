"""Data Ingestion / Collector (Section 7.1, item 2). Consumer side of
ADL-18's thread + queue producer-consumer pattern."""

from aiperfopt.ingestion.consumer import IngestionConsumer

__all__ = ["IngestionConsumer"]
