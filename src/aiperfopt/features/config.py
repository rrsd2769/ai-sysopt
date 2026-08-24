"""
Configuration for Feature Engineering (Section 7.1, item 4).

Kept as an explicit, small dataclass — like every other subsystem's
settings — rather than threading the full ``AppSettings`` object through
:mod:`aiperfopt.features.engine`, so the feature computation functions
stay testable in isolation (NFR-14.1) and don't acquire an implicit
dependency on unrelated configuration sections.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Parameters governing lag/rolling/rate feature computation.

    Attributes
    ----------
    lookback_windows_seconds:
        Window widths (seconds) used for both lag lookups and rolling
        statistics. Defaults are aligned with the two forecast horizons
        (Section 10.2: 30-60s short, 2-5min medium) — a 30s window feeds
        short-horizon forecasting features, 120s feeds medium-horizon
        trend features. The pipeline wiring (Phase 4+) is expected to
        derive these from ``AppSettings.forecasting`` rather than callers
        hardcoding a second copy of the horizon values.
    tolerance_seconds:
        Maximum allowed gap between a requested lag point (e.g. "30s ago")
        and the nearest actual sample for that lookup to be considered
        valid (FR-4.3). If the nearest sample is farther away than this,
        the lag feature is ``None`` rather than a stale value.
    max_gap_seconds:
        Maximum time between two consecutive raw samples for a
        counter-metric rate (bytes/sec) to be computed between them. A
        larger gap means the "rate" would be averaged over an unknown
        amount of missed activity, which is not a meaningful
        instantaneous rate — the resulting rate feature is ``None``.
    gauge_metrics:
        Metrics used as-is (lag/rolling statistics computed directly on
        the raw value).
    counter_metrics:
        Cumulative OS counters. These are rate-transformed
        (bytes/sec-style derivative) before any lag/rolling statistics
        are computed on them — see the module docstring in
        :mod:`aiperfopt.features.engine`.
    """

    lookback_windows_seconds: tuple[int, ...] = (30, 120)
    tolerance_seconds: float = 5.0
    max_gap_seconds: float = 30.0
    gauge_metrics: tuple[str, ...] = ("cpu_percent", "mem_percent", "swap_percent", "load_avg_1")
    counter_metrics: tuple[str, ...] = (
        "disk_read_bytes",
        "disk_write_bytes",
        "net_bytes_sent",
        "net_bytes_recv",
    )

    @property
    def min_history_seconds(self) -> float:
        """Minimum trailing window of raw data needed to compute every feature.

        Callers reading a live window from storage (e.g.
        :func:`aiperfopt.features.engine.compute_live_feature_vector`)
        should request at least this much history, or lag/rolling
        features for the largest window will legitimately come back as
        ``None`` for lack of data rather than a real gap.
        """
        return max(self.lookback_windows_seconds) + self.tolerance_seconds


FEATURE_SET_VERSION = "v1"
