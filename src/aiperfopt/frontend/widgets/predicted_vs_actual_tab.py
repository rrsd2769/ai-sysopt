"""
Controller widget for FR-13.3's comparison view: lets the user pick a
metric and a lookback window, fetches the join from
``GET /reporting/predicted-vs-actual`` (Phase 10's own endpoint) in the
background (see ``workers.py``), and feeds the result to
``PredictedVsActualChart``.
"""

from __future__ import annotations

import time

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from aiperfopt.applog import get_logger
from aiperfopt.frontend.api_client import ApiClient, ApiError
from aiperfopt.frontend.widgets.predicted_vs_actual import PredictedVsActualChart
from aiperfopt.frontend.workers import run_in_background

logger = get_logger(__name__)

# Metrics forecasting actually trains against (aiperfopt.forecasting.dataset);
# a small, curated subset for the picker rather than every raw_system_metrics
# column, since not every column is a meaningful forecasting target.
_METRIC_CHOICES = ["cpu_percent", "mem_percent", "swap_percent", "load_avg_1"]

# aiperfopt.forecasting.training_pipeline trains exactly these two horizons
# per metric. Leaving horizon unfiltered used to join *both* into one
# series -- two forecasts made at the same tick but targeting different
# future moments, with typically very different accuracy (medium-horizon
# MAE is usually several times short-horizon's), interleaved by
# target_timestamp. That produces a jagged, sawtooth-looking line with no
# single coherent story, not a real data problem. Defaulting to one
# horizon at a time is the fix; "short" first since it's the more accurate,
# more demo-relevant of the two.
_HORIZON_CHOICES = ["short", "medium"]


class PredictedVsActualTab(QWidget):
    def __init__(self, api_client: ApiClient, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api_client = api_client

        outer = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Metric:"))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems(_METRIC_CHOICES)
        controls.addWidget(self._metric_combo)

        controls.addWidget(QLabel("Horizon:"))
        self._horizon_combo = QComboBox()
        self._horizon_combo.addItems(_HORIZON_CHOICES)
        controls.addWidget(self._horizon_combo)

        controls.addWidget(QLabel("Lookback (minutes):"))
        self._lookback_spin = QSpinBox()
        self._lookback_spin.setRange(1, 24 * 60)
        self._lookback_spin.setValue(15)
        controls.addWidget(self._lookback_spin)

        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(self.refresh)
        controls.addWidget(self._refresh_button)
        controls.addStretch(1)
        outer.addLayout(controls)

        self._status_label = QLabel("")
        outer.addWidget(self._status_label)

        self._chart = PredictedVsActualChart(_METRIC_CHOICES[0], _HORIZON_CHOICES[0])
        outer.addWidget(self._chart)

    def refresh(self) -> None:
        metric_name = self._metric_combo.currentText()
        horizon_label = self._horizon_combo.currentText()
        lookback_seconds = self._lookback_spin.value() * 60
        self._chart.set_metric_name(metric_name, horizon_label)
        self._status_label.setText("Loading...")
        self._refresh_button.setEnabled(False)

        end = time.time()
        start = end - lookback_seconds

        def _call():
            return self._api_client.get_predicted_vs_actual(
                metric_name, start=start, end=end, horizon_label=horizon_label
            )

        run_in_background(_call, on_success=self._on_loaded, on_error=self._on_error)

    def _on_loaded(self, points: list[dict]) -> None:
        self._refresh_button.setEnabled(True)
        self._chart.update_points(points)
        self._status_label.setText(f"{len(points)} points" if points else "No predictions in this window yet.")

    def _on_error(self, exc: Exception) -> None:
        self._refresh_button.setEnabled(True)
        detail = exc.detail if isinstance(exc, ApiError) else str(exc)
        self._status_label.setText(f"Failed to load: {detail}")
        logger.warning("predicted-vs-actual refresh failed", extra={"error": str(exc)})
