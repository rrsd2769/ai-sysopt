"""
The distinct informational notification on significant
anomaly detection — separate from any contention prediction, and never
routed through the Recommendation Engine (the "Unusual Pattern
(Anomaly-Flagged)" taxonomy entry).

This is the first real caller of ``app_logs``/``insert_log_event`` —
The schema module docstring flagged that table as existing for
"significant pipeline events... explicitly persisted... not a
logging.Handler mirroring every line," and an anomaly notification is
exactly that: infrequent, meaningful, worth being queryable later
(Reporting) without parsing log files.
"""

from __future__ import annotations

from sqlalchemy import Engine

from aiperfopt.applog import get_logger
from aiperfopt.storage.records import LogEvent
from aiperfopt.storage.writer import insert_log_event

logger = get_logger(__name__)


def emit_anomaly_notification_if_significant(
    engine: Engine, anomaly_score: float, threshold: float, timestamp: float
) -> bool:
    """Persist a distinct notification if ``anomaly_score`` meets ``threshold``.

    Returns whether a notification was emitted, so a caller building a
    live response payload knows whether to surface one to the user this
    cycle.
    """
    if anomaly_score < threshold:
        return False

    insert_log_event(
        engine,
        LogEvent(
            timestamp=timestamp,
            level="WARNING",
            logger="aiperfopt.anomaly",
            message=(
                "Unusual system behavior detected. Prediction confidence has been "
                "reduced because the current workload differs significantly from "
                "previously observed patterns."
            ),
            context={"anomaly_score": anomaly_score, "threshold": threshold},
        ),
    )
    logger.warning(
        "Significant anomaly detected", extra={"anomaly_score": anomaly_score, "threshold": threshold}
    )
    return True
