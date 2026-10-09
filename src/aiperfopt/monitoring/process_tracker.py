"""
Process-level polling: per-process metrics plus launch/termination
detection, with the process-exclusion privacy mechanism applied at the source.

Correctness note on CPU percent
--------------------------------
``psutil.Process.cpu_percent(interval=None)`` reports usage *since the
previous call on that same Process object*. Constructing a fresh
``Process`` every poll cycle (e.g. via ``psutil.process_iter()`` each
time) would make every reading meaningless (psutil defines the first
call's return value as 0.0, or a lifetime-average with no useful
resolution). :class:`ProcessTracker` instead keeps a ``pid -> Process``
cache alive across polls: a newly-seen pid gets its first
``cpu_percent()`` call "primed" (result discarded) in the same cycle its
launch is detected, and every subsequent poll of that pid returns a real
interval-based delta.

This cache is also exactly the data structure that makes launch/
termination detection a cheap set-diff: pids present now but not
in the cache are launches; pids in the cache but no longer present are
terminations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import psutil

from aiperfopt.applog import get_logger
from aiperfopt.storage.records import ProcessEvent, ProcessMetricAggregate, ProcessMetricSample

logger = get_logger(__name__)

# Default storage bound: top-N by CPU union top-N by memory RSS
# are individually stored; used when a caller doesn't pass its own
# top_n (e.g. direct ProcessTracker() construction in tests). Production
# callers should pass MonitoringSettings.process_capture_top_n instead of
# relying on this default.
DEFAULT_CAPTURE_TOP_N = 20


@dataclass
class ProcessPollResult:
    metrics: list[ProcessMetricSample] = field(default_factory=list)
    events: list[ProcessEvent] = field(default_factory=list)
    aggregate: ProcessMetricAggregate | None = None


def bound_process_metrics(
    metrics: list[ProcessMetricSample], top_n: int, timestamp: float
) -> tuple[list[ProcessMetricSample], ProcessMetricAggregate]:
    """Split ``metrics`` into the individually-stored top-N and one aggregate row.

    Kept individually: the union of the top ``top_n`` processes by
    ``cpu_percent`` and the top ``top_n`` by ``memory_rss`` -- this
    guarantees every feature that only needs the top-K by one of those two
    axes (e.g. "the single busiest process") is exact, never approximated
    by the truncation. Everything else is folded into one
    :class:`ProcessMetricAggregate` row so per-poll storage stays bounded
    at roughly ``2 * top_n + 1`` rows regardless of how many processes are
    actually running, while sums/counts over *all* processes (not just
    the top-N) can still be reconstructed exactly by adding the aggregate
    back in (see ``aiperfopt.features.process_features``).
    """
    if len(metrics) <= top_n * 2:
        # Nothing to truncate -- still emit a zero aggregate row so every
        # poll cycle produces exactly one.
        return list(metrics), ProcessMetricAggregate(
            timestamp=timestamp, other_count=0, other_cpu_sum=0.0, other_mem_rss_sum=0
        )

    by_cpu = sorted(metrics, key=lambda m: m.cpu_percent, reverse=True)[:top_n]
    by_mem = sorted(metrics, key=lambda m: m.memory_rss or 0, reverse=True)[:top_n]
    kept_pids = {m.pid for m in by_cpu} | {m.pid for m in by_mem}

    kept = [m for m in metrics if m.pid in kept_pids]
    others = [m for m in metrics if m.pid not in kept_pids]

    aggregate = ProcessMetricAggregate(
        timestamp=timestamp,
        other_count=len(others),
        other_cpu_sum=sum(m.cpu_percent for m in others),
        other_mem_rss_sum=sum(m.memory_rss or 0 for m in others),
    )
    return kept, aggregate


class ProcessTracker:
    """Stateful per-process poller. One instance per Monitoring Engine."""

    def __init__(
        self,
        excluded_process_names: frozenset[str] = frozenset(),
        capture_top_n: int = DEFAULT_CAPTURE_TOP_N,
    ) -> None:
        self._processes: dict[int, psutil.Process] = {}
        self._excluded = excluded_process_names
        self._capture_top_n = capture_top_n
        # The very first poll establishes a baseline of already-running
        # processes; those are not "launches" (launches mean *new*
        # events observed during monitoring, not a dump of everything
        # already running when the engine starts).
        self._baseline_established = False

    def _is_excluded(self, name: str) -> bool:
        # Case-insensitive exact match against configured names (Section
        # 13's mechanism is presented to the user as a simple exclusion
        # list, not a pattern language — keeping matching simple here
        # keeps the user-facing behavior predictable).
        return name.lower() in self._excluded

    def poll(self) -> ProcessPollResult:
        """Poll all currently-running processes once.

        Returns metrics for every non-excluded live process, plus launch
        events for newly-seen pids and termination events for pids that
        vanished since the last poll. Excluded processes never appear in
        the returned metrics *or* events, satisfying the "exclude
        specific processes from historical storage, analytics,
        prediction, and reporting" — exclusion happens before anything is
        even queued, so no downstream stage ever sees them.
        """
        timestamp = time.time()
        result = ProcessPollResult()

        current_pids = set(psutil.pids())
        previous_pids = set(self._processes.keys())

        # --- Terminations: cached pids no longer present ---
        for gone_pid in previous_pids - current_pids:
            proc = self._processes.pop(gone_pid)
            name = getattr(proc, "_aiperfopt_name", None) or "unknown"
            if not self._is_excluded(name):
                result.events.append(
                    ProcessEvent(timestamp=timestamp, pid=gone_pid, name=name, event_type="terminate")
                )

        # --- Live processes: launches + metrics ---
        for pid in current_pids:
            is_new = pid not in self._processes
            try:
                proc = self._processes.get(pid) or psutil.Process(pid)
                name = proc.name()
                # Cache the name on the Process object itself so a later
                # termination event (where the process object can no
                # longer be queried) still knows what to call it.
                proc._aiperfopt_name = name  # type: ignore[attr-defined]

                if is_new:
                    self._processes[pid] = proc
                    proc.cpu_percent(interval=None)  # prime the counter
                    if self._baseline_established and not self._is_excluded(name):
                        result.events.append(
                            ProcessEvent(
                                timestamp=timestamp, pid=pid, name=name, event_type="launch"
                            )
                        )
                    # A freshly-primed process has no meaningful
                    # cpu_percent yet this cycle; it will report a real
                    # delta starting next poll.
                    continue

                if self._is_excluded(name):
                    continue

                cpu_percent = proc.cpu_percent(interval=None)
                mem_info = proc.memory_info()
                sample = ProcessMetricSample(
                    timestamp=timestamp,
                    pid=pid,
                    name=name,
                    cpu_percent=cpu_percent,
                    memory_rss=mem_info.rss,
                    memory_vms=mem_info.vms,
                    memory_percent=proc.memory_percent(),
                    num_threads=proc.num_threads(),
                    priority=proc.nice(),
                    status=proc.status(),
                    create_time=proc.create_time(),
                )
                result.metrics.append(sample)

            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied) as exc:
                # A single process disappearing or being unreadable mid-scan
                # is routine, not an engine-level failure (the
                # per-source degradation philosophy applied at process
                # granularity). Drop it from the cache so it doesn't leak.
                self._processes.pop(pid, None)
                logger.debug(
                    "Process became unavailable during poll",
                    extra={"pid": pid, "error": str(exc)},
                )

        self._baseline_established = True
        result.metrics, result.aggregate = bound_process_metrics(
            result.metrics, self._capture_top_n, timestamp
        )
        return result

    @property
    def tracked_process_count(self) -> int:
        return len(self._processes)
