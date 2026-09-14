from __future__ import annotations

import pytest

from aiperfopt.frontend.widgets.live_chart import LiveTrendChart

pytestmark = pytest.mark.behavioural

_STATE = {
    "timestamp": 1000.0,
    "latest_metrics": {"cpu_percent": 42.0, "mem_percent": 60.0},
    "forecasts": [
        {"metric_name": "cpu_percent", "target_timestamp": 1010.0, "predicted_value": 55.0, "confidence": 0.8},
        {"metric_name": "mem_percent", "target_timestamp": 1010.0, "predicted_value": 61.0, "confidence": None},
    ],
}


def test_actual_history_accumulates_across_ticks(qtbot):
    chart = LiveTrendChart()
    qtbot.addWidget(chart)

    chart.update_state(_STATE)
    chart.update_state(dict(_STATE, timestamp=1002.0, latest_metrics={"cpu_percent": 44.0, "mem_percent": 61.0}))

    assert chart._actual_series["cpu_percent"].count() == 2
    assert chart._actual_series["mem_percent"].count() == 2


def test_forecast_draws_a_two_point_segment_from_now(qtbot):
    chart = LiveTrendChart()
    qtbot.addWidget(chart)
    chart.update_state(_STATE)

    cpu_forecast = chart._forecast_series["cpu_percent"]
    assert cpu_forecast.count() == 2
    assert cpu_forecast.at(0).x() == 1000.0 * 1000
    assert cpu_forecast.at(0).y() == 42.0
    assert cpu_forecast.at(1).x() == 1010.0 * 1000
    assert cpu_forecast.at(1).y() == 55.0


def test_missing_forecast_clears_forecast_series(qtbot):
    chart = LiveTrendChart()
    qtbot.addWidget(chart)
    chart.update_state(_STATE)
    assert chart._forecast_series["cpu_percent"].count() == 2

    chart.update_state(dict(_STATE, timestamp=1002.0, forecasts=None))
    assert chart._forecast_series["cpu_percent"].count() == 0


def test_no_latest_metrics_leaves_history_untouched(qtbot):
    chart = LiveTrendChart()
    qtbot.addWidget(chart)
    chart.update_state(_STATE)
    chart.update_state({"timestamp": 1002.0, "latest_metrics": None, "forecasts": None})

    assert chart._actual_series["cpu_percent"].count() == 1


def test_confidence_band_widens_when_confidence_is_none(qtbot):
    chart = LiveTrendChart()
    qtbot.addWidget(chart)
    chart.update_state(_STATE)

    # cpu_percent has confidence=0.8 (narrow band); mem_percent has
    # confidence=None, treated as a mid-point default (wider band).
    cpu_half_width = chart._band_upper["cpu_percent"].at(1).y() - 55.0
    mem_half_width = chart._band_upper["mem_percent"].at(1).y() - 61.0
    assert mem_half_width > cpu_half_width


def test_y_axis_zooms_to_a_narrow_band_instead_of_staying_fixed_at_full_range(qtbot):
    """A metric sitting in a tight band (e.g. memory idling around 60%) used to
    render as a visually flat line on a fixed 0-100 axis. The axis should
    zoom to the data instead, so real variation stays visible regardless of
    the absolute level it's happening at."""
    chart = LiveTrendChart()
    qtbot.addWidget(chart)

    narrow_state = {
        "timestamp": 1000.0,
        "latest_metrics": {"cpu_percent": 61.0, "mem_percent": 62.0},
        "forecasts": None,
    }
    chart.update_state(narrow_state)
    chart.update_state(dict(narrow_state, timestamp=1002.0, latest_metrics={"cpu_percent": 63.0, "mem_percent": 64.0}))

    lo, hi = chart._axis_y.min(), chart._axis_y.max()
    assert hi - lo < 100.0
    assert lo < 61.0
    assert hi > 64.0


def test_y_axis_has_a_floor_span_when_the_machine_is_genuinely_idle(qtbot):
    """A completely flat metric shouldn't zoom the axis down to pixel noise."""
    chart = LiveTrendChart()
    qtbot.addWidget(chart)

    flat_state = {"timestamp": 1000.0, "latest_metrics": {"cpu_percent": 5.0, "mem_percent": 5.0}, "forecasts": None}
    chart.update_state(flat_state)
    chart.update_state(dict(flat_state, timestamp=1002.0))

    assert chart._axis_y.max() - chart._axis_y.min() >= 10.0
