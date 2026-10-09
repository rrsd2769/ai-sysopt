"""
Recording and portable serialization of a real monitoring window.

A :class:`RecordedSession` is a snapshot of exactly what Data Storage
accumulated over some real time window -- the four raw-sample tables
Monitoring/Ingestion populate: system metrics,
process metrics, process events, and the "everything else" process
aggregate. Nothing here computes or interprets anything; recording is a
pure read-and-serialize step so a replayed session, once its timestamps
are shifted (see :mod:`aiperfopt.replay.player`), is bit-for-bit the real
data a real pressure event actually produced.

Kept out of :mod:`aiperfopt.replay.player` because "capture a window" and
"play it back with shifted timestamps into a different database" are
independently useful and independently testable -- a session file, once
recorded, can be replayed any number of times at any speed without
re-touching the primary database.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from sqlalchemy import Engine

from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import ReplayError
from aiperfopt.storage.reader import (
    get_process_events_range,
    get_process_metric_aggregates_range,
    get_process_metrics_range,
    get_system_metrics_range,
)
from aiperfopt.storage.records import (
    ProcessEvent,
    ProcessMetricAggregate,
    ProcessMetricSample,
    SystemMetricSample,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class RecordedSession:
    """A portable, self-contained recording of one real time window.

    ``start``/``end`` are the actual span covered by ``system_metrics``
    (not the requested window) -- the span replay anchors against, since
    that is the data that will actually be replayed.
    """

    recorded_at: float
    start: float
    end: float
    system_metrics: tuple[SystemMetricSample, ...] = field(default_factory=tuple)
    process_metrics: tuple[ProcessMetricSample, ...] = field(default_factory=tuple)
    process_events: tuple[ProcessEvent, ...] = field(default_factory=tuple)
    process_metric_aggregates: tuple[ProcessMetricAggregate, ...] = field(default_factory=tuple)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end - self.start)

    def shifted(self, delta_seconds: float) -> "RecordedSession":
        """A new session with every timestamp moved by ``delta_seconds``.

        A pure linear shift, never a rescale -- every record's relative
        spacing to every other record is preserved exactly, which is what
        keeps feature vectors computed against a replayed session
        identical to the ones computed against the original recording
        regardless of playback ``speed`` (see
        :func:`aiperfopt.replay.player.replay_session`). ``id`` is cleared
        on every record: a rewritten row is not the same row as the one
        recorded from the primary database.
        """
        return replace(
            self,
            start=self.start + delta_seconds,
            end=self.end + delta_seconds,
            system_metrics=tuple(
                replace(s, timestamp=s.timestamp + delta_seconds, id=None) for s in self.system_metrics
            ),
            process_metrics=tuple(
                replace(s, timestamp=s.timestamp + delta_seconds, id=None) for s in self.process_metrics
            ),
            process_events=tuple(
                replace(e, timestamp=e.timestamp + delta_seconds, id=None) for e in self.process_events
            ),
            process_metric_aggregates=tuple(
                replace(a, timestamp=a.timestamp + delta_seconds, id=None)
                for a in self.process_metric_aggregates
            ),
        )


def record_session(engine: Engine, minutes: float, *, end: float | None = None) -> RecordedSession:
    """Capture the trailing ``minutes`` of real accumulated history from ``engine``.

    Raises :class:`ReplayError` if there is no system-metric data in the
    window at all -- a session with nothing in it has no meaningful
    ``start``/``end`` span for replay to anchor against, and almost
    certainly means the collector was not running during the requested
    window.
    """
    reference = time.time() if end is None else end
    window_start = reference - minutes * 60.0

    system_metrics = get_system_metrics_range(engine, start=window_start, end=reference)
    if not system_metrics:
        raise ReplayError(
            f"No system metrics found in the last {minutes:g} minute(s) -- is the "
            "collector running? ('aiperfopt collect' / 'aiperfopt run')",
            recoverable=False,
        )

    actual_start = system_metrics[0].timestamp
    actual_end = system_metrics[-1].timestamp

    session = RecordedSession(
        recorded_at=time.time(),
        start=actual_start,
        end=actual_end,
        system_metrics=tuple(system_metrics),
        process_metrics=tuple(get_process_metrics_range(engine, start=actual_start, end=actual_end)),
        process_events=tuple(get_process_events_range(engine, start=actual_start, end=actual_end)),
        process_metric_aggregates=tuple(
            get_process_metric_aggregates_range(engine, start=actual_start, end=actual_end)
        ),
    )
    logger.info(
        "Session recorded",
        extra={
            "system_metric_count": len(session.system_metrics),
            "duration_seconds": session.duration_seconds,
        },
    )
    return session


def save_session(session: RecordedSession, path: Path) -> None:
    """Serialize ``session`` to a portable JSON file."""
    payload = {
        "recorded_at": session.recorded_at,
        "start": session.start,
        "end": session.end,
        "system_metrics": [asdict(s) for s in session.system_metrics],
        "process_metrics": [asdict(s) for s in session.process_metrics],
        "process_events": [asdict(e) for e in session.process_events],
        "process_metric_aggregates": [asdict(a) for a in session.process_metric_aggregates],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def load_session(path: Path) -> RecordedSession:
    """Deserialize a session file written by :func:`save_session`."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return RecordedSession(
        recorded_at=payload["recorded_at"],
        start=payload["start"],
        end=payload["end"],
        system_metrics=tuple(SystemMetricSample(**row) for row in payload["system_metrics"]),
        process_metrics=tuple(ProcessMetricSample(**row) for row in payload["process_metrics"]),
        process_events=tuple(ProcessEvent(**row) for row in payload["process_events"]),
        process_metric_aggregates=tuple(
            ProcessMetricAggregate(**row) for row in payload["process_metric_aggregates"]
        ),
    )
