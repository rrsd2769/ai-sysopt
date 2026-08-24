"""
Developer Mode panel (Section 4): technical users, evaluators, the
developer. Surfaces fields already present in the same ``LiveStateOut``
payload ``dashboard.py`` renders -- process statistics, contention
internals, feature-attribution (SHAP/native importance), and model
versions -- rather than fetching anything extra (see the package
docstring for why there is no separate "detailed" endpoint to call).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from aiperfopt.frontend.formatting import format_confidence, format_percent, format_probability


class DeveloperPanel(QWidget):
    """Renders the technical detail behind a ``LiveStateOut``-shaped dict."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)

        content = QWidget()
        content_layout = QVBoxLayout(content)

        self._model_version_label = QLabel("Active model version: --")
        content_layout.addWidget(self._model_version_label)

        self._feature_vector_tree = _labeled_tree("Feature vector", ["Feature", "Value"])
        content_layout.addWidget(self._feature_vector_tree.parent_frame)

        self._contention_frame = QFrame()
        self._contention_frame.setFrameShape(QFrame.Shape.StyledPanel)
        contention_layout = QVBoxLayout(self._contention_frame)
        contention_layout.addWidget(_section_title("Contention prediction"))
        self._contention_summary = QLabel("--")
        self._contention_summary.setWordWrap(True)
        contention_layout.addWidget(self._contention_summary)
        self._contributing_metrics_tree = _tree(["Metric", "Contribution"])
        contention_layout.addWidget(self._contributing_metrics_tree)
        self._contributing_processes_tree = _tree(["PID", "Process", "Value"])
        contention_layout.addWidget(self._contributing_processes_tree)
        content_layout.addWidget(self._contention_frame)

        self._diagnosis_frame = QFrame()
        self._diagnosis_frame.setFrameShape(QFrame.Shape.StyledPanel)
        diagnosis_layout = QVBoxLayout(self._diagnosis_frame)
        diagnosis_layout.addWidget(_section_title("Diagnosis / explainable AI"))
        self._diagnosis_technical = QLabel("--")
        self._diagnosis_technical.setWordWrap(True)
        diagnosis_layout.addWidget(self._diagnosis_technical)
        self._diagnosis_meta = QLabel("--")
        diagnosis_layout.addWidget(self._diagnosis_meta)
        self._forecast_explanations_tree = _tree(["Metric", "Tier", "Top feature", "Attribution"])
        diagnosis_layout.addWidget(self._forecast_explanations_tree)
        content_layout.addWidget(self._diagnosis_frame)

        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def update_state(self, state: dict[str, Any]) -> None:
        self._model_version_label.setText(
            f"Active model version: {state.get('active_model_version') or '--'}"
        )

        feature_vector = state.get("feature_vector") or {}
        self._feature_vector_tree.tree.clear()
        for name, value in sorted(feature_vector.items()):
            QTreeWidgetItem(self._feature_vector_tree.tree, [name, "--" if value is None else f"{value:.4f}"])

        self._update_contention(state.get("contention"))
        self._update_diagnosis(state.get("diagnosis"))

    def _update_contention(self, contention: dict[str, Any] | None) -> None:
        self._contributing_metrics_tree.clear()
        self._contributing_processes_tree.clear()
        if contention is None:
            self._contention_summary.setText("No contention prediction this cycle.")
            return

        self._contention_summary.setText(
            f"Probability: {format_probability(contention.get('probability'))}  |  "
            f"Threshold: {format_probability(contention.get('threshold_used'))}  |  "
            f"Triggered: {contention.get('triggered')}  |  "
            f"Model: {contention.get('model_version', '--')}  |  "
            f"Anomaly score: {contention.get('anomaly_score')}"
        )
        for entry in contention.get("contributing_metrics") or []:
            QTreeWidgetItem(
                self._contributing_metrics_tree,
                [str(entry.get("metric_name", "")), f"{entry.get('contribution', 0.0):.4f}"],
            )
        for entry in contention.get("contributing_processes") or []:
            QTreeWidgetItem(
                self._contributing_processes_tree,
                [str(entry.get("pid", "")), str(entry.get("name", "")), f"{entry.get('value', 0.0):.2f}"],
            )

    def _update_diagnosis(self, diagnosis: dict[str, Any] | None) -> None:
        self._forecast_explanations_tree.clear()
        if diagnosis is None:
            self._diagnosis_technical.setText("No diagnosis this cycle.")
            self._diagnosis_meta.setText("")
            return

        self._diagnosis_technical.setText(diagnosis.get("technical_explanation") or "")
        self._diagnosis_meta.setText(
            f"Category: {diagnosis.get('category', '--')}  |  "
            f"Actionable: {diagnosis.get('is_actionable')}  |  "
            f"Confidence: {format_confidence(diagnosis.get('confidence'))}  |  "
            f"Explanation tier: {diagnosis.get('explanation_tier', '--')}  |  "
            f"Trigger: {diagnosis.get('trigger_reason', '--')}"
        )
        for explanation in diagnosis.get("forecast_explanations") or []:
            top_features = explanation.get("top_features") or []
            top = top_features[0] if top_features else None
            QTreeWidgetItem(
                self._forecast_explanations_tree,
                [
                    str(explanation.get("metric_name", "")),
                    str(explanation.get("tier", "")),
                    "" if top is None else str(top.get("feature_name", "")),
                    "" if top is None else f"{top.get('value', 0.0):.4f}",
                ],
            )


class _LabeledTree:
    def __init__(self, parent_frame: QFrame, tree: QTreeWidget) -> None:
        self.parent_frame = parent_frame
        self.tree = tree


def _section_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("font-weight: 600; font-size: 14px;")
    return label


def _tree(headers: list[str]) -> QTreeWidget:
    tree = QTreeWidget()
    tree.setHeaderLabels(headers)
    tree.setRootIsDecorated(False)
    return tree


def _labeled_tree(title: str, headers: list[str]) -> _LabeledTree:
    frame = QFrame()
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    layout = QVBoxLayout(frame)
    layout.addWidget(_section_title(title))
    tree = _tree(headers)
    layout.addWidget(tree)
    return _LabeledTree(frame, tree)
