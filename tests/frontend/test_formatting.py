from __future__ import annotations

from aiperfopt.diagnosis.taxonomy import DiagnosisCategory
from aiperfopt.frontend.formatting import (
    confidence_tier,
    format_confidence,
    format_duration_short,
    format_percent,
    format_probability,
    format_timestamp,
    humanize_bytes,
    humanize_category,
    readiness_banner_text,
)


def test_humanize_category_covers_every_real_taxonomy_value():
    """Every category the backend can actually emit must have a real label,
    not silently fall through to the raw-string fallback."""
    for category in DiagnosisCategory:
        label = humanize_category(category.value)
        assert label != category.value
        assert label[0].isupper() or label[0] == "O"  # "One process..." etc.


def test_humanize_category_falls_back_gracefully_for_unknown_value():
    assert humanize_category("some_future_category") == "Some future category"


def test_humanize_bytes():
    assert humanize_bytes(None) == "--"
    assert humanize_bytes(0) == "0.0 B"
    assert humanize_bytes(1536) == "1.5 KB"
    assert humanize_bytes(1024 * 1024 * 3) == "3.0 MB"


def test_format_percent():
    assert format_percent(None) == "--"
    assert format_percent(53.256) == "53.3%"
    assert format_percent(53.256, decimals=0) == "53%"


def test_format_confidence_and_probability_are_fractions_to_percent():
    assert format_confidence(None) == "--"
    assert format_confidence(0.876) == "88%"
    assert format_probability(0.4321) == "43.2%"


def test_format_timestamp_none():
    assert format_timestamp(None) == "--"
    assert ":" in format_timestamp(1700000000.0)


def test_confidence_tier_thresholds():
    assert confidence_tier(None) == "unknown"
    assert confidence_tier(0.9) == "high"
    assert confidence_tier(0.6) == "medium"
    assert confidence_tier(0.1) == "low"
    assert confidence_tier(0.75) == "high"
    assert confidence_tier(0.5) == "medium"


def test_format_duration_short():
    assert format_duration_short(45.0) == "45s"
    assert format_duration_short(135.0) == "2m15s"
    assert format_duration_short(0.0) == "0s"


def test_readiness_banner_text_for_each_state():
    assert readiness_banner_text("no_data", "detail", 0.0, 300.0) == "No data yet"
    assert readiness_banner_text("collecting", "detail", 30.0, 300.0) == "Collecting - 30s of 5m00s"
    assert readiness_banner_text("needs_training", "detail", 300.0, 300.0) == "No model trained"
    assert readiness_banner_text("ready", "detail", 300.0, 300.0) == "Live"
    assert readiness_banner_text("replay", "detail", 300.0, 300.0) == "REPLAY"


def test_readiness_banner_text_falls_back_to_detail_for_unknown_state():
    assert readiness_banner_text("some_future_state", "a detail string", 0.0, 0.0) == "a detail string"
