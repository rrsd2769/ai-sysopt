"""
Replays a :class:`~aiperfopt.replay.session.RecordedSession` into a
separate database, standing in for the live collector so a recorded real
pressure event can be reproduced on demand (Phase 8, "the demonstration
mechanism"). The live inference loop (:mod:`aiperfopt.api.live_state`)
then runs over the replay database unmodified -- nothing about it knows
or needs to know its data came from a replay rather than a live collector.

Design: a shift, never a rescale
---------------------------------
Timestamps are rewritten by a pure linear shift
(:meth:`~aiperfopt.replay.session.RecordedSession.shifted`), never
rescaled -- every record's relative spacing to every other record is
identical to the original recording regardless of playback ``speed``.
This is what the plan's own success criterion demands ("a 10x replay
still yields the same feature values as real time"): Feature Engineering
(:mod:`aiperfopt.features.engine`) computes every lag/rolling-window
feature purely from timestamp *deltas*, never from wall-clock time or
absolute time-of-day (see ``aiperfopt.common.timeutils`` -- ``_dt`` is
only ever used as a pandas time index for relative windowing), so a pure
shift leaves every feature value unchanged no matter how far the whole
session is moved.

``speed`` therefore controls exactly one thing: how fast, in real
wall-clock seconds, the insertion loop below advances through the
(unrescaled) rewritten data -- a playback-pacing knob for the live demo
(Phase 9's chart), never a transformation of the data itself. Pacing is
anchored to system-metric samples (Monitoring's steady heartbeat,
Section 7.1 item 1) the same way the real
:class:`~aiperfopt.monitoring.engine.MonitoringEngine` produces one
system sample per poll cycle -- process-level rows that arrived before a
given system sample (but after the previous one) are written alongside
it.
"""

from __future__ import annotations

import bisect
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from sqlalchemy import Engine, select

from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import ReplayError
from aiperfopt.readiness import STALE_AFTER_SECONDS
from aiperfopt.replay.session import RecordedSession
from aiperfopt.storage import schema
from aiperfopt.storage.reader import get_recent_system_metrics
from aiperfopt.storage.writer import (
    insert_process_events_batch,
    insert_process_metric_aggregates_batch,
    insert_process_metric_samples_batch,
    insert_system_metric_samples_batch,
)

logger = get_logger(__name__)


def copy_model_metadata(primary_engine: Engine, replay_engine: Engine) -> None:
    """Copy every ``evaluations`` and ``model_deployments`` row from
    ``primary_engine`` into ``replay_engine``.

    Raw system/process samples are the only thing the insertion loop
    below writes into a fresh replay database. But the live-inference
    loop reconstructs each deployed forecaster from persisted evaluation
    history (``resolve_deployed_forecasters``, keyed by model version),
    not from the model artifact files alone -- without this, a server
    pointed at the replay database finds the shared
    ``active_model_version.json`` pointer (a plain file, not
    database-scoped) but no evaluation rows to rebuild that version's
    forecasters from, and every live-inference tick fails with "no usable
    short-horizon forecasters" even though the model artifacts on disk
    are perfectly fine. ``model_deployments`` is copied too so the
    Retraining tab's deployment history isn't misleadingly empty during a
    replay demo. Both tables are small (one row per training/deployment
    event, not per sample), so copying every row rather than filtering to
    one model version keeps this simple and still correct if the active
    version changes between recording and replaying.
    """
    with primary_engine.connect() as src, replay_engine.begin() as dst:
        for table in (schema.evaluations, schema.model_deployments):
            rows = [dict(row._mapping) for row in src.execute(select(table))]
            if rows:
                dst.execute(table.insert(), rows)


def is_monitoring_active(engine: Engine) -> bool:
    """True if ``engine`` has a system-metric sample fresher than
    :data:`aiperfopt.readiness.STALE_AFTER_SECONDS`.

    This is the same staleness heuristic
    :func:`aiperfopt.readiness.assess_readiness` already uses to answer
    the inverse question ("has monitoring stopped?"), reused here so the
    project has exactly one definition of "is monitoring running" instead
    of two that could drift apart.
    """
    recent = get_recent_system_metrics(engine, window_seconds=STALE_AFTER_SECONDS)
    if not recent:
        return False
    return (time.time() - recent[-1].timestamp) <= STALE_AFTER_SECONDS


def assert_monitoring_not_active(engine: Engine) -> None:
    """Raise :class:`ReplayError` if ``engine`` looks like a live, running collector.

    The plan's own constraint: "Replay must refuse to run if the
    monitoring pipeline is active, to prevent real and replayed samples
    interleaving." ``engine`` here is always the *primary* database --
    replay itself never writes there.
    """
    if is_monitoring_active(engine):
        raise ReplayError(
            "Refusing to replay: the monitoring pipeline appears to be active on the "
            "primary database (a system-metric sample landed within the last "
            f"{STALE_AFTER_SECONDS:.0f}s). Stop it first -- replay and live monitoring "
            "must never write to the same database concurrently.",
            recoverable=False,
        )


@dataclass(frozen=True)
class ReplaySummary:
    system_metric_count: int
    process_metric_count: int
    process_event_count: int
    process_metric_aggregate_count: int
    start: float
    end: float


def replay_session(
    session: RecordedSession,
    replay_engine: Engine,
    *,
    speed: float = 1.0,
    primary_engine: Engine | None = None,
    anchor_end: float | None = None,
    on_tick: Callable[[int, int], None] | None = None,
) -> ReplaySummary:
    """Replay ``session`` into ``replay_engine``, never the primary database.

    Parameters
    ----------
    primary_engine:
        When given, checked with :func:`assert_monitoring_not_active`
        before anything is written. Omit only from tests that have no
        primary database to guard.
    anchor_end:
        Real epoch second the session's last sample should land on --
        "the first sample maps to now minus session length" from the
        plan. Defaults to ``time.time()``; overridable for deterministic
        tests.
    speed:
        Playback rate. 1.0 replays across the same real time the session
        originally spanned; 10.0 finishes ten times faster. Must be > 0.
    on_tick:
        Optional progress callback invoked after each system-metric tick
        is written, as ``(ticks_done, ticks_total)``.
    """
    if speed <= 0:
        raise ValueError("speed must be > 0")
    if primary_engine is not None:
        assert_monitoring_not_active(primary_engine)
    if not session.system_metrics:
        raise ReplayError("Session has no system metrics to replay.", recoverable=False)

    anchor_end = time.time() if anchor_end is None else anchor_end
    rewritten = session.shifted(anchor_end - session.end)

    ticks = _bucket_by_system_tick(rewritten)
    total_ticks = len(ticks)
    previous_ts = rewritten.system_metrics[0].timestamp

    for index, (system_sample, process_metrics, process_events, aggregates) in enumerate(ticks):
        wait_seconds = max(0.0, (system_sample.timestamp - previous_ts) / speed)
        time.sleep(wait_seconds)
        previous_ts = system_sample.timestamp

        insert_system_metric_samples_batch(replay_engine, [system_sample])
        if process_metrics:
            insert_process_metric_samples_batch(replay_engine, process_metrics)
        if process_events:
            insert_process_events_batch(replay_engine, process_events)
        if aggregates:
            insert_process_metric_aggregates_batch(replay_engine, aggregates)

        if on_tick is not None:
            on_tick(index + 1, total_ticks)

    logger.info(
        "Replay complete",
        extra={
            "system_metric_count": len(rewritten.system_metrics),
            "process_metric_count": len(rewritten.process_metrics),
            "process_event_count": len(rewritten.process_events),
            "process_metric_aggregate_count": len(rewritten.process_metric_aggregates),
            "speed": speed,
        },
    )
    return ReplaySummary(
        system_metric_count=len(rewritten.system_metrics),
        process_metric_count=len(rewritten.process_metrics),
        process_event_count=len(rewritten.process_events),
        process_metric_aggregate_count=len(rewritten.process_metric_aggregates),
        start=rewritten.start,
        end=rewritten.end,
    )


def _bucket_by_system_tick(
    session: RecordedSession,
) -> list[tuple]:
    """Group process-level rows onto the system-metric tick they arrived by.

    Every process/event/aggregate row is attached to the first
    system-metric sample at or after its own timestamp -- matching how
    the real ``MonitoringEngine``'s two independent polling threads feed
    one shared queue, process-level rows never predate the system sample
    they would have been drained alongside by more than one poll cycle.
    Anything after the very last system sample is attached to it.
    """
    system_timestamps = [s.timestamp for s in session.system_metrics]
    last_index = len(system_timestamps) - 1

    def _bucket_index(ts: float) -> int:
        return min(bisect.bisect_left(system_timestamps, ts), last_index)

    process_buckets: list[list] = [[] for _ in system_timestamps]
    event_buckets: list[list] = [[] for _ in system_timestamps]
    aggregate_buckets: list[list] = [[] for _ in system_timestamps]

    def _fill(items: Sequence, buckets: list[list]) -> None:
        for item in items:
            buckets[_bucket_index(item.timestamp)].append(item)

    _fill(session.process_metrics, process_buckets)
    _fill(session.process_events, event_buckets)
    _fill(session.process_metric_aggregates, aggregate_buckets)

    return list(zip(session.system_metrics, process_buckets, event_buckets, aggregate_buckets))
