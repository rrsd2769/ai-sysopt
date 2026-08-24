from __future__ import annotations

from aiperfopt.frontend.widgets.predicted_vs_actual import PredictedVsActualChart


def test_constructs_with_metric_name(qtbot):
    chart = PredictedVsActualChart("cpu_percent")
    qtbot.addWidget(chart)
    assert chart.metric_name == "cpu_percent"


def test_update_points_populates_both_series(qtbot):
    chart = PredictedVsActualChart("cpu_percent")
    qtbot.addWidget(chart)
    points = [
        {"target_timestamp": 1000.0, "predicted_value": 50.0, "actual_value": 48.0},
        {"target_timestamp": 1010.0, "predicted_value": 55.0, "actual_value": 53.0},
        {"target_timestamp": 1020.0, "predicted_value": 60.0, "actual_value": None},
    ]
    chart.update_points(points)
    assert chart._predicted_series.count() == 3
    # The third point has no actual_value yet -- skipped, not plotted as 0.
    assert chart._actual_series.count() == 2


def test_update_points_with_empty_list_clears_series(qtbot):
    chart = PredictedVsActualChart("cpu_percent")
    qtbot.addWidget(chart)
    chart.update_points([{"target_timestamp": 1000.0, "predicted_value": 1.0, "actual_value": 1.0}])
    assert chart._predicted_series.count() == 1
    chart.update_points([])
    assert chart._predicted_series.count() == 0
    assert chart._actual_series.count() == 0


def test_set_metric_name_updates_title(qtbot):
    chart = PredictedVsActualChart()
    qtbot.addWidget(chart)
    chart.set_metric_name("mem_percent")
    assert chart.metric_name == "mem_percent"
    assert "mem_percent" in chart._chart.title()
