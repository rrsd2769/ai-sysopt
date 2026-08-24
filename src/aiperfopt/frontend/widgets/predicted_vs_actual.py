"""
The predicted-vs-actual comparison view required by FR-13.3, rendering
Section 6's flagship "wow moment": predicted values plotted alongside
actual observed values over time, sourced from the Phase 10 reporting
endpoint (``GET /reporting/predicted-vs-actual``) rather than any
client-side joining logic (see that endpoint's module docstring for why
the join lives server-side).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCharts import QChart, QChartView, QDateTimeAxis, QLineSeries, QValueAxis
from PySide6.QtCore import QDateTime, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QVBoxLayout, QWidget


class PredictedVsActualChart(QWidget):
    """Draws two series -- predicted and actual -- for one metric over time.

    Call :meth:`update_points` with the list of dicts
    ``ApiClient.get_predicted_vs_actual(...)`` returns
    (``PredictedVsActualPointOut``'s JSON shape); points whose
    ``actual_value`` is ``None`` (not yet observed, per that endpoint's
    documented behavior) are simply skipped from the actual-value series
    rather than plotted as zero.
    """

    def __init__(self, metric_name: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.metric_name = metric_name

        self._chart = QChart()
        self._chart.setTitle(f"Predicted vs. actual: {metric_name}" if metric_name else "Predicted vs. actual")
        self._chart.legend().setVisible(True)

        self._predicted_series = QLineSeries()
        self._predicted_series.setName("Predicted")
        self._actual_series = QLineSeries()
        self._actual_series.setName("Actual")
        self._chart.addSeries(self._predicted_series)
        self._chart.addSeries(self._actual_series)

        self._axis_x = QDateTimeAxis()
        self._axis_x.setFormat("HH:mm:ss")
        self._axis_x.setTitleText("Time")
        self._chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)
        self._predicted_series.attachAxis(self._axis_x)
        self._actual_series.attachAxis(self._axis_x)

        self._axis_y = QValueAxis()
        self._axis_y.setTitleText("Value")
        self._chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)
        self._predicted_series.attachAxis(self._axis_y)
        self._actual_series.attachAxis(self._axis_y)

        view = QChartView(self._chart)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)

        layout = QVBoxLayout(self)
        layout.addWidget(view)

    def set_metric_name(self, metric_name: str) -> None:
        self.metric_name = metric_name
        self._chart.setTitle(f"Predicted vs. actual: {metric_name}")

    def update_points(self, points: list[dict[str, Any]]) -> None:
        self._predicted_series.clear()
        self._actual_series.clear()
        if not points:
            return

        values: list[float] = []
        for point in points:
            ts_ms = int(point["target_timestamp"] * 1000)
            predicted = float(point["predicted_value"])
            self._predicted_series.append(ts_ms, predicted)
            values.append(predicted)
            actual = point.get("actual_value")
            if actual is not None:
                self._actual_series.append(ts_ms, float(actual))
                values.append(float(actual))

        first_ts_ms = int(points[0]["target_timestamp"] * 1000)
        last_ts_ms = int(points[-1]["target_timestamp"] * 1000)
        self._axis_x.setRange(
            QDateTime.fromMSecsSinceEpoch(first_ts_ms), QDateTime.fromMSecsSinceEpoch(last_ts_ms)
        )

        if values:
            lo, hi = min(values), max(values)
            padding = max((hi - lo) * 0.1, 1.0)
            self._axis_y.setRange(lo - padding, hi + padding)
