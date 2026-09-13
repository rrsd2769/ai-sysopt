from __future__ import annotations

from aiperfopt.frontend.widgets.contention_strip import ContentionStrip


def test_appends_a_point_when_contention_present(qtbot):
    strip = ContentionStrip()
    qtbot.addWidget(strip)

    state = {
        "timestamp": 1000.0,
        "contention": {"probability": 0.6, "threshold_used": 0.5, "triggered": True},
    }
    strip.update_state(state)

    assert strip._probability_series.count() == 1
    assert strip._threshold_series.count() == 2  # a flat two-point line across the window
    assert strip._threshold == 0.5


def test_missing_contention_leaves_history_and_threshold_unchanged(qtbot):
    strip = ContentionStrip()
    qtbot.addWidget(strip)
    strip.update_state({
        "timestamp": 1000.0,
        "contention": {"probability": 0.6, "threshold_used": 0.5, "triggered": True},
    })
    strip.update_state({"timestamp": 1002.0, "contention": None})

    assert strip._probability_series.count() == 1
    assert strip._threshold == 0.5


def test_no_contention_ever_seen_draws_nothing(qtbot):
    strip = ContentionStrip()
    qtbot.addWidget(strip)
    strip.update_state({"timestamp": 1000.0, "contention": None})

    assert strip._probability_series.count() == 0
    assert strip._threshold_series.count() == 0
