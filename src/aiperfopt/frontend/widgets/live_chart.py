"""
Rolling actual-vs-forecast trend chart for the dashboard (the
flagship "predictive" moment): a rolling window of actual CPU and
memory as solid lines, with the current forecast drawn as a dashed
continuation past "now" plus a confidence band.

Fed one ``LiveStateOut``-shaped dict per tick via :meth:`update_state`,
same pattern as ``DashboardWidget`` -- this widget holds no fetching logic
of its own. Unlike ``PredictedVsActualChart`` (which renders an
already-complete history from the reporting endpoint), this widget has to
build its own rolling history client-side: a ``LiveStateOut`` snapshot
carries only the current tick's ``latest_metrics``/``forecasts``, never a
window of history (see ``aiperfopt.api.live_state``'s module docstring --
Phases 3-8 are all one-shot computations; this loop is the first
continuously-cadenced one, and its wire format reflects that).
"""

from __future__ import annotations

from collections import deque
from typing import Any

from PySide6.QtCharts import QAreaSeries, QChart, QChartView, QDateTimeAxis, QLineSeries, QValueAxis
from PySide6.QtCore import QDateTime, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QVBoxLayout, QWidget

# Rolling window length. Not time-based (the live-inference tick interval
# is a runtime setting, api.live_inference_interval_seconds) -- a fixed
# point count keeps memory/render cost bounded regardless of how fast
# ticks arrive.
_MAX_POINTS = 300

# Half-width, in percentage points, applied to the confidence band at
# zero confidence (see _redraw_forecast for why this is a display
# heuristic rather than a statistical interval).
_MAX_BAND_HALF_WIDTH = 15.0

# A fixed 0-100 y-axis makes real variation invisible whenever a metric
# spends most of its time in a narrow band (e.g. memory sitting at
# 60-70%, or CPU idling under 10% with only occasional spikes) -- both
# lines end up compressed into a thin, visually "flat" strip even though
# the underlying data is moving. Auto-scaling to the currently-visible
# range fixes that, at the cost of the axis itself changing over time;
# a minimum span keeps it from over-zooming into pure sensor noise when
# the machine is genuinely idle and flat.
_MIN_Y_SPAN = 10.0
_Y_PADDING_FRACTION = 0.15

_METRIC_LABELS: dict[str, str] = {"cpu_percent": "CPU", "mem_percent": "Memory"}
_METRIC_COLORS: dict[str, QColor] = {"cpu_percent": QColor("#1976d2"), "mem_percent": QColor("#6a1b9a")}


class LiveTrendChart(QWidget):
    """Rolling actual CPU/memory with the current forecast as a dashed lead-in.

    Each tick's forecast is drawn as a short two-point dashed segment from
    "now" (the latest actual value) to that forecast's target timestamp --
    not an accumulating history of past forecasts. That is what lets a
    viewer see the flagship moment directly: the solid line arriving where
    the *previous* tick's dashed line said it would.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._actual_history: dict[str, deque[tuple[int, float]]] = {
            metric: deque(maxlen=_MAX_POINTS) for metric in _METRIC_LABELS
        }
        # The current tick's forecast/band extent per metric, so the y-axis
        # rescale below can include "where the dashed line and its band
        # currently reach" -- not just observed history -- without the
        # axis clipping the forecast off-screen the moment it exceeds the
        # actual data's own range.
        self._latest_forecast_range: dict[str, tuple[float, float] | None] = dict.fromkeys(_METRIC_LABELS)

        self._chart = QChart()
        self._chart.setTitle("Live trend: actual vs. forecast")
        self._chart.legend().setVisible(True)

        self._actual_series: dict[str, QLineSeries] = {}
        self._forecast_series: dict[str, QLineSeries] = {}
        self._band_upper: dict[str, QLineSeries] = {}
        self._band_lower: dict[str, QLineSeries] = {}
        all_series = []

        for metric_name, label in _METRIC_LABELS.items():
            color = _METRIC_COLORS[metric_name]

            upper = QLineSeries()
            lower = QLineSeries()
            band = QAreaSeries(upper, lower)
            band.setPen(QPen(Qt.PenStyle.NoPen))
            band.setBrush(QColor(color.red(), color.green(), color.blue(), 40))
            band.setName(f"{label} confidence")
            self._band_upper[metric_name] = upper
            self._band_lower[metric_name] = lower

            forecast = QLineSeries()
            forecast.setName(f"{label} forecast")
            pen = QPen(color)
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(2)
            forecast.setPen(pen)
            self._forecast_series[metric_name] = forecast

            actual = QLineSeries()
            actual.setName(f"{label} actual")
            actual.setPen(QPen(color, 2))
            self._actual_series[metric_name] = actual

            for series in (band, forecast, actual):
                self._chart.addSeries(series)
                all_series.append(series)

        self._axis_x = QDateTimeAxis()
        self._axis_x.setFormat("HH:mm:ss")
        self._axis_x.setTitleText("Time")
        self._chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)

        self._axis_y = QValueAxis()
        self._axis_y.setTitleText("Percent")
        self._axis_y.setRange(0.0, _MIN_Y_SPAN)
        self._chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)

        for series in all_series:
            series.attachAxis(self._axis_x)
            series.attachAxis(self._axis_y)

        view = QChartView(self._chart)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view)

    def update_state(self, state: dict[str, Any]) -> None:
        latest = state.get("latest_metrics")
        if latest is None:
            return

        now_ms = int(state["timestamp"] * 1000)
        forecasts_by_metric = {f["metric_name"]: f for f in (state.get("forecasts") or [])}

        max_ms = now_ms
        for metric_name in _METRIC_LABELS:
            current_value = latest.get(metric_name)
            if current_value is not None:
                self._actual_history[metric_name].append((now_ms, float(current_value)))
            self._redraw_actual(metric_name)
            forecast = forecasts_by_metric.get(metric_name)
            self._redraw_forecast(metric_name, now_ms, current_value, forecast)
            if forecast is not None:
                max_ms = max(max_ms, int(forecast["target_timestamp"] * 1000))

        self._rescale_x_axis(max_ms)
        self._rescale_y_axis()

    def _redraw_actual(self, metric_name: str) -> None:
        series = self._actual_series[metric_name]
        series.clear()
        for ts_ms, value in self._actual_history[metric_name]:
            series.append(ts_ms, value)

    def _redraw_forecast(
        self, metric_name: str, now_ms: int, current_value: float | None, forecast: dict[str, Any] | None
    ) -> None:
        forecast_series = self._forecast_series[metric_name]
        upper = self._band_upper[metric_name]
        lower = self._band_lower[metric_name]
        forecast_series.clear()
        upper.clear()
        lower.clear()

        if forecast is None or current_value is None:
            self._latest_forecast_range[metric_name] = None
            return

        target_ms = int(forecast["target_timestamp"] * 1000)
        predicted = float(forecast["predicted_value"])
        forecast_series.append(now_ms, float(current_value))
        forecast_series.append(target_ms, predicted)

        # `confidence` is a 0-1 model-derived scalar ("uncalibrated"),
        # not an interval, so there is no true upper/lower bound to plot.
        # This widens the band linearly as confidence drops toward 0, up to
        # a fixed cap, so a low-confidence forecast visibly hedges more
        # than a high-confidence one -- a display heuristic, not a
        # statistical interval.
        confidence = forecast.get("confidence")
        half_width = (1.0 - confidence if confidence is not None else 0.5) * _MAX_BAND_HALF_WIDTH
        band_upper = min(100.0, predicted + half_width)
        band_lower = max(0.0, predicted - half_width)
        upper.append(now_ms, float(current_value))
        upper.append(target_ms, band_upper)
        lower.append(now_ms, float(current_value))
        lower.append(target_ms, band_lower)
        self._latest_forecast_range[metric_name] = (band_lower, band_upper)

    def _rescale_y_axis(self) -> None:
        """Zoom the shared y-axis to what's actually on screen right now.

        A fixed 0-100 range is honest but makes a metric that spends most
        of its time in a narrow band (memory idling at 60-70%, CPU mostly
        under 10%) look like a flat line regardless of real movement.
        Rescaling to the current window's actual+forecast extent (with a
        floor so a genuinely idle machine doesn't zoom into pixel noise)
        keeps the same variation visible whatever level it's happening at.
        """
        values: list[float] = [v for history in self._actual_history.values() for _, v in history]
        for bounds in self._latest_forecast_range.values():
            if bounds is not None:
                values.extend(bounds)
        if not values:
            return

        lo, hi = min(values), max(values)
        span = hi - lo
        if span < _MIN_Y_SPAN:
            pad = (_MIN_Y_SPAN - span) / 2.0
            lo, hi = lo - pad, hi + pad
        else:
            padding = span * _Y_PADDING_FRACTION
            lo, hi = lo - padding, hi + padding
        self._axis_y.setRange(max(0.0, lo), min(100.0, hi))

    def _rescale_x_axis(self, max_ms: int) -> None:
        all_ts = [ts for history in self._actual_history.values() for ts, _ in history]
        if not all_ts:
            return
        self._axis_x.setRange(QDateTime.fromMSecsSinceEpoch(min(all_ts)), QDateTime.fromMSecsSinceEpoch(max_ms))
