"""
Top-level window (FR-13.1). Assembles the widgets built above around one
``ApiClient`` (REST) and one ``LiveWebSocketClient`` (``/ws/live``
streaming), and implements the General/Developer mode toggle (Section 4)
as a pure rendering switch -- both modes are fed the exact same
``LiveStateOut`` payload (see the package docstring).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, QTimer

from aiperfopt.applog import get_logger
from aiperfopt.frontend.api_client import ApiClient, ApiError
from aiperfopt.frontend.config import FrontendSettings
from aiperfopt.frontend.live_client import LiveWebSocketClient
from aiperfopt.frontend.widgets.dashboard import DashboardWidget
from aiperfopt.frontend.widgets.developer_panel import DeveloperPanel
from aiperfopt.frontend.widgets.predicted_vs_actual_tab import PredictedVsActualTab
from aiperfopt.frontend.widgets.reporting_tab import ReportingTab
from aiperfopt.frontend.widgets.retraining_panel import RetrainingPanel
from aiperfopt.frontend.workers import run_in_background

logger = get_logger(__name__)

_MODE_GENERAL = "General Mode"
_MODE_DEVELOPER = "Developer Mode"

# Readiness (Phase 9 F9) is polled on its own cadence via REST, separate
# from the WebSocket LiveStateOut stream -- it answers "can this system
# predict right now, and why not" (aiperfopt.readiness), not part of any
# single live-inference tick's payload.
_READINESS_POLL_INTERVAL_MS = 3000


class MainWindow(QMainWindow):
    def __init__(
        self,
        api_client: ApiClient | None = None,
        live_client: LiveWebSocketClient | None = None,
        settings: FrontendSettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI System Performance Optimizer")

        settings = settings or FrontendSettings()
        self.api_client = api_client or ApiClient(settings=settings)
        self.live_client = live_client or LiveWebSocketClient(settings=settings)
        self.live_client.state_received.connect(self._on_live_state)
        self.live_client.connection_changed.connect(self._on_connection_changed)

        self._readiness_timer = QTimer(self)
        self._readiness_timer.setInterval(_READINESS_POLL_INTERVAL_MS)
        self._readiness_timer.timeout.connect(self._poll_readiness)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        outer.addLayout(self._build_top_bar())

        self._live_split = QSplitter(Qt.Orientation.Horizontal)
        self.dashboard = DashboardWidget()
        self.developer_panel = DeveloperPanel()
        self._live_split.addWidget(self.dashboard)
        self._live_split.addWidget(self.developer_panel)

        self.predicted_vs_actual_tab = PredictedVsActualTab(self.api_client)
        self.reporting_tab = ReportingTab(self.api_client)
        self.retraining_panel = RetrainingPanel(self.api_client)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._live_split, "Live")
        self._tabs.addTab(self.predicted_vs_actual_tab, "Predicted vs. Actual")
        self._tabs.addTab(self.reporting_tab, "Reporting")
        self._tabs.addTab(self.retraining_panel, "Retraining")
        outer.addWidget(self._tabs)

        self.set_mode(_MODE_GENERAL)

    # --- top bar -------------------------------------------------------------

    def _build_top_bar(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self._connection_label = QLabel("Disconnected")
        self._connection_label.setStyleSheet("color: #b71c1c; font-weight: 600;")
        row.addWidget(self._connection_label)

        row.addWidget(QLabel("View:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([_MODE_GENERAL, _MODE_DEVELOPER])
        self._mode_combo.currentTextChanged.connect(self.set_mode)
        row.addWidget(self._mode_combo)

        self._monitoring_button = QPushButton("Start monitoring")
        self._monitoring_button.clicked.connect(self._toggle_monitoring)
        row.addWidget(self._monitoring_button)

        self._live_inference_button = QPushButton("Start live inference")
        self._live_inference_button.clicked.connect(self._toggle_live_inference)
        row.addWidget(self._live_inference_button)

        row.addStretch(1)
        return row

    # --- mode toggle -----------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        is_developer = mode == _MODE_DEVELOPER
        self.developer_panel.setVisible(is_developer)
        if self._mode_combo.currentText() != mode:
            self._mode_combo.setCurrentText(mode)
        # Retraining is a Developer Mode concern (Phase 8/9's manual,
        # human-confirmed deployment machinery isn't something a General
        # Mode user needs or should casually stumble into).
        self._tabs.setTabVisible(self._tabs.indexOf(self.retraining_panel), is_developer)

    # --- live state plumbing ----------------------------------------------------

    def _on_live_state(self, state: dict[str, Any]) -> None:
        self.dashboard.update_state(state)
        self.developer_panel.update_state(state)

    def _on_connection_changed(self, connected: bool) -> None:
        if connected:
            self._connection_label.setText("Connected")
            self._connection_label.setStyleSheet("color: #1e4620; font-weight: 600;")
        else:
            self._connection_label.setText("Disconnected")
            self._connection_label.setStyleSheet("color: #b71c1c; font-weight: 600;")

    # --- monitoring / live inference controls -----------------------------------

    def _toggle_monitoring(self) -> None:
        def _call():
            status = self.api_client.monitoring_status()
            return self.api_client.stop_monitoring() if status["running"] else self.api_client.start_monitoring()

        run_in_background(_call, on_success=self._on_monitoring_toggled, on_error=self._on_control_error)

    def _on_monitoring_toggled(self, result: dict) -> None:
        self._monitoring_button.setText("Stop monitoring" if result["running"] else "Start monitoring")

    def _toggle_live_inference(self) -> None:
        def _call():
            status = self.api_client.live_status()
            return (
                self.api_client.stop_live_inference()
                if status["running"]
                else self.api_client.start_live_inference()
            )

        run_in_background(_call, on_success=self._on_live_inference_toggled, on_error=self._on_control_error)

    def _on_live_inference_toggled(self, result: dict) -> None:
        self._live_inference_button.setText(
            "Stop live inference" if result["running"] else "Start live inference"
        )

    def _on_control_error(self, exc: Exception) -> None:
        detail = exc.detail if isinstance(exc, ApiError) else str(exc)
        logger.warning("main window control action failed", extra={"error": detail})

    # --- readiness polling -------------------------------------------------------

    def _poll_readiness(self) -> None:
        run_in_background(self.api_client.get_readiness, on_success=self.dashboard.update_readiness)

    # --- lifecycle --------------------------------------------------------------

    def start(self) -> None:
        """Begin streaming live state and reflect current backend control state."""
        self.live_client.start()
        run_in_background(self.api_client.monitoring_status, on_success=self._on_monitoring_toggled)
        run_in_background(self.api_client.live_status, on_success=self._on_live_inference_toggled)
        self.retraining_panel.refresh_active_version()
        self._poll_readiness()
        self._readiness_timer.start()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        self._readiness_timer.stop()
        self.live_client.stop()
        self.api_client.close()
        super().closeEvent(event)
