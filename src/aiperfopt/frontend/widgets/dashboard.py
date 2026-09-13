"""
General Mode dashboard (Section 4): the everyday-user view. Emphasizes
overall system health, plain-language explanations, and simple
recommendations -- no raw feature vectors, SHAP output, or model
versions here (those live in ``developer_panel.py``); this widget and
that one render the *same* ``LiveStateOut``-shaped dict, differing only
in which fields they choose to show (see the package docstring).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from aiperfopt.frontend.formatting import (
    confidence_tier,
    format_confidence,
    format_percent,
    format_probability,
    humanize_category,
    readiness_banner_text,
)
from aiperfopt.frontend.widgets.contention_strip import ContentionStrip
from aiperfopt.frontend.widgets.live_chart import LiveTrendChart

_TIER_COLORS = {
    "high": "#2e7d32",
    "medium": "#ef6c00",
    "low": "#c62828",
    "unknown": "#616161",
}

# Background/text colors per `GET /readiness` state (Phase 9 F9), keyed by
# ReadinessState's value. "replay" is deliberately its own distinct color
# (not a variant of "ready"'s green) -- the plan requires an *unmistakable*
# indicator that this is replayed, not live, data.
_READINESS_STYLES = {
    "no_data": ("#eeeeee", "#424242"),
    "collecting": ("#fff3cd", "#664d03"),
    "needs_training": ("#ffe0b2", "#7a4100"),
    "ready": ("#e6f4ea", "#1e4620"),
    "replay": ("#ede7f6", "#4527a0"),
}
_ERROR_STYLE = ("#fdecea", "#b71c1c")


class _MetricTile(QFrame):
    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        self._value_label = QLabel("--")
        self._value_label.setStyleSheet("font-size: 22px; font-weight: 600;")
        self._title_label = QLabel(label)
        self._title_label.setStyleSheet("color: #666;")
        layout.addWidget(self._value_label)
        layout.addWidget(self._title_label)

    def set_value(self, text: str) -> None:
        self._value_label.setText(text)


class DashboardWidget(QWidget):
    """Renders a ``LiveStateOut``-shaped dict for a non-technical user.

    Call :meth:`update_state` with each decoded WebSocket payload (or a
    REST ``/live/state`` response) -- this widget holds no fetching logic
    of its own, matching FR-13.2.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)

        self._last_error: str | None = None
        self._last_readiness: dict[str, Any] | None = None

        self._status_banner = QLabel("Waiting for live data...")
        self._status_banner.setStyleSheet(
            "padding: 10px; border-radius: 6px; background: #eeeeee; font-weight: 600;"
        )
        outer.addWidget(self._status_banner)

        tiles_row = QHBoxLayout()
        self._cpu_tile = _MetricTile("CPU")
        self._mem_tile = _MetricTile("Memory")
        self._contention_tile = _MetricTile("Contention risk")
        for tile in (self._cpu_tile, self._mem_tile, self._contention_tile):
            tiles_row.addWidget(tile)
        outer.addLayout(tiles_row)

        self._live_chart = LiveTrendChart()
        outer.addWidget(self._live_chart)

        self._contention_strip = ContentionStrip()
        outer.addWidget(self._contention_strip)

        self._anomaly_banner = QLabel()
        self._anomaly_banner.setWordWrap(True)
        self._anomaly_banner.setStyleSheet(
            "padding: 8px; border-radius: 6px; background: #fff3cd; color: #664d03;"
        )
        self._anomaly_banner.hide()
        outer.addWidget(self._anomaly_banner)

        diagnosis_frame = QFrame()
        diagnosis_frame.setFrameShape(QFrame.Shape.StyledPanel)
        diagnosis_layout = QVBoxLayout(diagnosis_frame)
        self._diagnosis_title = QLabel("No issue detected right now")
        self._diagnosis_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        self._diagnosis_title.setWordWrap(True)
        self._diagnosis_body = QLabel("")
        self._diagnosis_body.setWordWrap(True)
        self._diagnosis_confidence = QLabel("")
        diagnosis_layout.addWidget(self._diagnosis_title)
        diagnosis_layout.addWidget(self._diagnosis_body)
        diagnosis_layout.addWidget(self._diagnosis_confidence)
        outer.addWidget(diagnosis_frame)

        recs_label = QLabel("Recommended actions")
        recs_label.setStyleSheet("font-weight: 600; margin-top: 8px;")
        outer.addWidget(recs_label)

        self._recs_container = QWidget()
        self._recs_layout = QVBoxLayout(self._recs_container)
        self._recs_layout.setContentsMargins(0, 0, 0, 0)
        self._recs_empty_label = QLabel("Nothing to act on right now.")
        self._recs_empty_label.setStyleSheet("color: #666;")
        self._recs_layout.addWidget(self._recs_empty_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._recs_container)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        outer.addWidget(scroll)

    # --- public API -----------------------------------------------------

    def update_state(self, state: dict[str, Any]) -> None:
        self._last_error = state.get("error")
        self._render_banner()

        self._live_chart.update_state(state)
        self._contention_strip.update_state(state)

        metrics = state.get("latest_metrics") or {}
        self._cpu_tile.set_value(format_percent(metrics.get("cpu_percent")))
        self._mem_tile.set_value(format_percent(metrics.get("mem_percent")))

        contention = state.get("contention")
        if contention is not None:
            self._contention_tile.set_value(format_probability(contention.get("probability")))
        else:
            self._contention_tile.set_value("--")

        if state.get("anomaly_notification"):
            self._anomaly_banner.setText(
                "Unusual system behavior detected. Prediction confidence has been "
                "reduced because the current workload differs from what's normally seen."
            )
            self._anomaly_banner.show()
        else:
            self._anomaly_banner.hide()

        self._update_diagnosis(state.get("diagnosis"))
        self._update_recommendation(state.get("recommendation"))

    def update_readiness(self, readiness: dict[str, Any]) -> None:
        """Feed one ``GET /readiness`` response (Phase 9 F9).

        Polled separately from the WebSocket ``LiveStateOut`` stream (see
        ``MainWindow``) since readiness -- can this system predict right
        now, and if not, why -- is not itself part of that payload; it
        answers a different question than any single live-inference tick
        does.
        """
        self._last_readiness = readiness
        self._render_banner()

    # --- internal ---------------------------------------------------------

    def _render_banner(self) -> None:
        if self._last_error:
            bg, fg = _ERROR_STYLE
            self._status_banner.setText(f"Something went wrong: {self._last_error}")
            self._status_banner.setStyleSheet(
                f"padding: 10px; border-radius: 6px; background: {bg}; color: {fg}; font-weight: 600;"
            )
            self._status_banner.setToolTip("")
            return

        if self._last_readiness is None:
            return  # keep the initial "Waiting for live data..." text until the first /readiness response

        readiness = self._last_readiness
        state = readiness.get("state", "")
        text = readiness_banner_text(
            state,
            readiness.get("detail", ""),
            readiness.get("history_seconds", 0.0),
            readiness.get("history_seconds_required", 0.0),
        )
        bg, fg = _READINESS_STYLES.get(state, _READINESS_STYLES["no_data"])
        self._status_banner.setText(text)
        self._status_banner.setStyleSheet(
            f"padding: 10px; border-radius: 6px; background: {bg}; color: {fg}; font-weight: 600;"
        )
        self._status_banner.setToolTip(readiness.get("detail", ""))

    def _update_diagnosis(self, diagnosis: dict[str, Any] | None) -> None:
        if diagnosis is None:
            self._diagnosis_title.setText("No issue detected right now")
            self._diagnosis_body.setText("")
            self._diagnosis_confidence.setText("")
            return

        self._diagnosis_title.setText(humanize_category(diagnosis.get("category", "")))
        self._diagnosis_body.setText(diagnosis.get("plain_language_explanation") or "")

        tier = confidence_tier(diagnosis.get("confidence"))
        color = _TIER_COLORS[tier]
        self._diagnosis_confidence.setText(
            f"Confidence: {format_confidence(diagnosis.get('confidence'))}"
        )
        self._diagnosis_confidence.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _update_recommendation(self, recommendation: dict[str, Any] | None) -> None:
        # Clear previous action cards.
        while self._recs_layout.count():
            item = self._recs_layout.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not self._recs_empty_label:
                widget.deleteLater()

        actions = (recommendation or {}).get("actions") or []
        if not actions:
            self._recs_layout.addWidget(self._recs_empty_label)
            self._recs_empty_label.show()
            return

        self._recs_empty_label.hide()
        for action in actions:
            card = QFrame()
            card.setFrameShape(QFrame.Shape.StyledPanel)
            card_layout = QVBoxLayout(card)
            title = QLabel(action.get("title", ""))
            title.setStyleSheet("font-weight: 600;")
            title.setWordWrap(True)
            description = QLabel(action.get("description", ""))
            description.setWordWrap(True)
            card_layout.addWidget(title)
            card_layout.addWidget(description)
            self._recs_layout.addWidget(card)
