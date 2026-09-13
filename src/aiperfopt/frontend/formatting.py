"""
Pure display-formatting helpers shared across widgets.

Deliberately Qt-free so these are trivially unit-testable without an
``QApplication``/offscreen platform plugin -- widgets import from here
rather than reimplementing formatting inline, keeping the same "one
source of truth" discipline the rest of the project applies to
production logic (e.g. Feature Engineering's FR-4.2).
"""

from __future__ import annotations

# Section 12's eight diagnosis categories, each with a short plain-language
# label for General Mode and an icon-ish glyph for quick visual scanning.
# This is *display* mapping only -- Diagnosis (Phase 6) already produces
# ``plain_language_explanation`` text; this is just the category chip/badge
# label, not a duplicate of that generated explanation.
_CATEGORY_LABELS: dict[str, str] = {
    "cpu_bound_single_process": "One process is using most of your CPU",
    "cpu_bound_multi_process_contention": "Several processes together are straining your CPU",
    "memory_pressure_gradual": "Memory usage is climbing steadily",
    "memory_pressure_sudden": "Memory usage just spiked sharply",
    "disk_io_bottleneck": "Disk activity is becoming a bottleneck",
    "network_saturation": "Network usage is approaching capacity",
    "cross_resource_contention": "Multiple resources are under pressure at once",
    "unusual_pattern_anomaly_flagged": "Unusual system behavior detected",
}


def humanize_category(category: str) -> str:
    """Map a ``DiagnosisRecord.category`` value to a General-Mode-friendly label.

    Falls back to a readable title-cased version of the raw category for
    any value not in the taxonomy snapshot above, so a future Section 12
    addition degrades gracefully instead of showing a KeyError or a raw
    enum string.
    """
    if category in _CATEGORY_LABELS:
        return _CATEGORY_LABELS[category]
    return category.replace("_", " ").strip().capitalize()


def humanize_bytes(num_bytes: float | None) -> str:
    """E.g. ``1536.0 -> "1.5 KB"``. Returns ``"--"`` for ``None``."""
    if num_bytes is None:
        return "--"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def format_percent(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "--"
    return f"{value:.{decimals}f}%"


def format_confidence(value: float | None) -> str:
    """Confidence scores are stored as [0, 1] fractions; render as a percentage."""
    if value is None:
        return "--"
    return f"{value * 100:.0f}%"


def format_probability(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.1f}%"


def format_timestamp(epoch_seconds: float | None) -> str:
    if epoch_seconds is None:
        return "--"
    import datetime

    return datetime.datetime.fromtimestamp(epoch_seconds).strftime("%H:%M:%S")


_READINESS_LABELS: dict[str, str] = {
    "no_data": "No data yet",
    "needs_training": "No model trained",
    "ready": "Live",
    "replay": "REPLAY",
}


def format_duration_short(seconds: float) -> str:
    """E.g. ``135.0 -> "2m15s"``, ``45.0 -> "45s"``."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def readiness_banner_text(state: str, detail: str, history_seconds: float, history_seconds_required: float) -> str:
    """Short banner text for a ``GET /readiness`` state (Phase 9 F9), the
    replacement for the previously-hardcoded "Live" banner
    (``DashboardWidget`` used to show green "Live" unconditionally,
    identical whether or not anything was actually being predicted).

    Deliberately short -- the banner is a one-line status strip, not a
    place for a full sentence; ``detail`` remains available (e.g. as a
    tooltip) for the full explanation. Falls back to ``detail`` itself for
    any state not in the table above, so an unrecognized future state
    degrades to something readable rather than a blank banner.
    """
    if state == "collecting":
        collected = format_duration_short(history_seconds)
        required = format_duration_short(history_seconds_required)
        return f"Collecting - {collected} of {required}"
    return _READINESS_LABELS.get(state, detail)


def confidence_tier(value: float | None) -> str:
    """A coarse label for confidence, used to pick a status color.

    Thresholds are deliberately conservative to echo Section 11's
    "false positives are more damaging than false negatives" stance --
    the UI should not paint something "High" confidence unless it really
    is, since an over-confident-looking display undermines the same
    trust-first priority the backend's decision thresholds are tuned
    for.
    """
    if value is None:
        return "unknown"
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "medium"
    return "low"
