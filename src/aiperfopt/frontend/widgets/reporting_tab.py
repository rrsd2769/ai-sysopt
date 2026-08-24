"""
Reporting tab (Section 7.1 item 14, FR-14.1/FR-14.2, Phase 11). Renders
``GET /reporting/summary``'s report -- system-metrics context,
prediction accuracy, contention, diagnosis, and recommendation
summaries -- for a chosen period. Available in both modes (unlike
Retraining, which is Developer-only): the report ties directly back to
the flagship pillars (ADL-22) a General Mode user cares about too (e.g.
"how accurate were the predictions this week"), it just doesn't expose
anything as technical as SHAP attribution or model versions.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from aiperfopt.applog import get_logger
from aiperfopt.frontend.api_client import ApiClient, ApiError
from aiperfopt.frontend.formatting import format_confidence, format_percent, format_probability, humanize_category
from aiperfopt.frontend.workers import run_in_background

logger = get_logger(__name__)


class ReportingTab(QWidget):
    def __init__(self, api_client: ApiClient, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api_client = api_client

        outer = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Period:"))
        self._period_combo = QComboBox()
        self._period_combo.addItems(["daily", "weekly"])
        controls.addWidget(self._period_combo)
        self._refresh_button = QPushButton("Generate report")
        self._refresh_button.clicked.connect(self.refresh)
        controls.addWidget(self._refresh_button)
        controls.addStretch(1)
        outer.addLayout(controls)

        self._status_label = QLabel("")
        outer.addWidget(self._status_label)

        content = QWidget()
        content_layout = QVBoxLayout(content)

        self._system_metrics_tree = _section("System metrics", ["Metric", "Avg", "Min", "Max", "Samples"], content_layout)
        self._prediction_accuracy_tree = _section(
            "Prediction accuracy", ["Metric", "Horizon", "MAE", "RMSE", "Directional accuracy", "Matched / total"],
            content_layout,
        )

        contention_box = QGroupBox("Contention")
        contention_layout = QVBoxLayout(contention_box)
        self._contention_label = QLabel("--")
        contention_layout.addWidget(self._contention_label)
        content_layout.addWidget(contention_box)

        diagnosis_box = QGroupBox("Diagnosis")
        diagnosis_layout = QVBoxLayout(diagnosis_box)
        self._diagnosis_label = QLabel("--")
        diagnosis_layout.addWidget(self._diagnosis_label)
        self._diagnosis_tree = _tree(["Category", "Count"])
        diagnosis_layout.addWidget(self._diagnosis_tree)
        content_layout.addWidget(diagnosis_box)

        recommendation_box = QGroupBox("Recommendations")
        recommendation_layout = QVBoxLayout(recommendation_box)
        self._recommendation_label = QLabel("--")
        recommendation_layout.addWidget(self._recommendation_label)
        content_layout.addWidget(recommendation_box)

        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def refresh(self) -> None:
        period = self._period_combo.currentText()
        self._status_label.setText("Generating report...")
        self._refresh_button.setEnabled(False)

        def _call():
            return self._api_client.get_report_summary(period=period)

        run_in_background(_call, on_success=self._on_report_ready, on_error=self._on_error)

    def _on_report_ready(self, report: dict[str, Any]) -> None:
        self._refresh_button.setEnabled(True)
        self._status_label.setText(
            f"Report window: {_fmt(report['start'])} -- {_fmt(report['end'])}"
        )

        self._system_metrics_tree.clear()
        for m in report["system_metrics"]:
            QTreeWidgetItem(
                self._system_metrics_tree,
                [
                    m["metric_name"], format_percent(m["avg"]) if m["avg"] is not None else "--",
                    "--" if m["minimum"] is None else f"{m['minimum']:.2f}",
                    "--" if m["maximum"] is None else f"{m['maximum']:.2f}",
                    str(m["sample_count"]),
                ],
            )

        self._prediction_accuracy_tree.clear()
        for p in report["prediction_accuracy"]:
            QTreeWidgetItem(
                self._prediction_accuracy_tree,
                [
                    p["metric_name"], p["horizon_label"],
                    "--" if p["mae"] is None else f"{p['mae']:.3f}",
                    "--" if p["rmse"] is None else f"{p['rmse']:.3f}",
                    "--" if p["directional_accuracy"] is None else format_probability(p["directional_accuracy"]),
                    f"{p['matched_sample_count']} / {p['prediction_count']}",
                ],
            )
        if not report["prediction_accuracy"]:
            self._prediction_accuracy_tree.addTopLevelItem(
                QTreeWidgetItem(["No predictions in this window yet.", "", "", "", "", ""])
            )

        contention = report["contention"]
        self._contention_label.setText(
            f"{contention['prediction_count']} predictions, "
            f"{contention['triggered_count']} triggered  |  "
            f"Avg probability: {format_probability(contention['avg_probability'])}"
        )

        diagnosis = report["diagnosis"]
        self._diagnosis_label.setText(
            f"{diagnosis['total_count']} diagnoses, {diagnosis['actionable_count']} actionable  |  "
            f"Avg confidence: {format_confidence(diagnosis['avg_confidence'])}"
        )
        self._diagnosis_tree.clear()
        for category, count in sorted(diagnosis["counts_by_category"].items()):
            if count == 0:
                continue
            QTreeWidgetItem(self._diagnosis_tree, [humanize_category(category), str(count)])

        recommendation = report["recommendation"]
        self._recommendation_label.setText(
            f"{recommendation['total_count']} recommendations, {recommendation['total_actions']} total actions"
        )

    def _on_error(self, exc: Exception) -> None:
        self._refresh_button.setEnabled(True)
        detail = exc.detail if isinstance(exc, ApiError) else str(exc)
        self._status_label.setText(f"Failed to generate report: {detail}")
        logger.warning("reporting tab refresh failed", extra={"error": str(exc)})


def _fmt(epoch_seconds: float) -> str:
    import datetime

    return datetime.datetime.fromtimestamp(epoch_seconds).strftime("%Y-%m-%d %H:%M")


def _tree(headers: list[str]) -> QTreeWidget:
    tree = QTreeWidget()
    tree.setHeaderLabels(headers)
    tree.setRootIsDecorated(False)
    return tree


def _section(title: str, headers: list[str], parent_layout: QVBoxLayout) -> QTreeWidget:
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    tree = _tree(headers)
    layout.addWidget(tree)
    parent_layout.addWidget(box)
    return tree
