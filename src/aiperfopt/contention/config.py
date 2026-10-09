"""
Configuration for Contention Prediction's training-label construction.

This mirrors :class:`aiperfopt.features.config.FeatureEngineeringConfig`'s
pattern deliberately: a plain dataclass, not a layered pydantic
``AppSettings`` section, because it governs *how training labels are
derived* (an ML/data-preparation concern analogous to Feature
Engineering's window/tolerance choices) rather than a runtime operational
parameter a deployed system needs to reconfigure without a retrain.
The *decision threshold*, by contrast, genuinely is an operational
parameter (used at inference time against an already-trained model's
output) and lives in ``AppSettings.contention`` accordingly — see
:mod:`aiperfopt.config.settings`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContentionLabelingConfig:
    """Per-metric thresholds defining "resource pressure severe enough to
    count as contention," used only to construct binary training labels.

    A row is labeled positive if *any* tracked metric's actual future
    value (at the short forecast horizon) meets or exceeds its threshold
    here. These map loosely to the diagnosis taxonomy triggering
    patterns (e.g. CPU-Bound categories trigger on forecasted CPU rise,
    Memory Pressure on forecasted memory saturation) without resolving to
    a specific taxonomy label — that resolution is Diagnosis's job,
    not Contention Prediction's.
    """

    value_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "cpu_percent": 70.0,
            "mem_percent": 85.0,
            "swap_percent": 50.0,
            "disk_read_bytes": 50_000.0,
            "disk_write_bytes": 50_000.0,
            "net_bytes_sent": 50_000.0,
            "net_bytes_recv": 50_000.0,
        }
    )
    tolerance_seconds: float = 5.0
