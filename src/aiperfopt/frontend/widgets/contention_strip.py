"""
Contention-probability strip for the dashboard (Phase 9 sub-part 3): a
slim rolling chart of predicted contention probability with the fitted
decision threshold marked, so a viewer can see the probability rise
*before* a spike rather than only learn about it after a diagnosis fires.

Fed the same ``LiveStateOut``-shaped dict as ``LiveTrendChart`` via
:meth:`update_state`, meant to sit directly underneath it and share its
rolling time window.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from PySide6.QtCharts import QChart, QChartView, QDateTimeAxis, QLineSeries, QValueAxis
from PySide6.QtCore import QDateTime, QMargins, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QVBoxLayout, QWidget

_MAX_POINTS = 300


class ContentionStrip(QWidget):
    """Rolling contention-probability history with the decision threshold marked.

    Only ticks that actually produced a ``contention`` prediction append a
    point -- a cycle that returned early (no active model, no trained
    classifier, no usable feature row; see
    ``LiveInferenceLoop._tick``'s early returns, all NFR-11.1 graceful
    degradation) leaves a gap in the line rather than plotting a
    fabricated zero.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(120)

        self._history: deque[tuple[int, float]] = deque(maxlen=_MAX_POINTS)
        self._threshold: float | None = None

        self._chart = QChart()
        self._chart.setTitle("Contention probability")
        self._chart.legend().setVisible(False)
        self._chart.setMargins(QMargins(4, 4, 4, 4))

        self._probability_series = QLineSeries()
        self._probability_series.setName("Probability")
        self._probability_series.setPen(QPen(QColor("#c62828"), 2))
        self._chart.addSeries(self._probability_series)

        self._threshold_series = QLineSeries()
        self._threshold_series.setName("Decision threshold")
        threshold_pen = QPen(QColor("#616161"))
        threshold_pen.setStyle(Qt.PenStyle.DashLine)
        self._threshold_series.setPen(threshold_pen)
        self._chart.addSeries(self._threshold_series)

        self._axis_x = QDateTimeAxis()
        self._axis_x.setFormat("HH:mm:ss")
        self._chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)

        self._axis_y = QValueAxis()
        self._axis_y.setRange(0.0, 1.0)
        self._chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)

        for series in (self._probability_series, self._threshold_series):
            series.attachAxis(self._axis_x)
            series.attachAxis(self._axis_y)

        view = QChartView(self._chart)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view)

    def update_state(self, state: dict[str, Any]) -> None:
        contention = state.get("contention")
        if contention is not None:
            ts_ms = int(state["timestamp"] * 1000)
            self._history.append((ts_ms, float(contention["probability"])))
            self._threshold = float(contention["threshold_used"])

        self._probability_series.clear()
        for ts_ms, probability in self._history:
            self._probability_series.append(ts_ms, probability)

        self._threshold_series.clear()
        if self._threshold is not None and self._history:
            first_ms = self._history[0][0]
            last_ms = self._history[-1][0]
            self._threshold_series.append(first_ms, self._threshold)
            self._threshold_series.append(last_ms, self._threshold)

        if self._history:
            self._axis_x.setRange(
                QDateTime.fromMSecsSinceEpoch(self._history[0][0]),
                QDateTime.fromMSecsSinceEpoch(self._history[-1][0]),
            )
