"""
Developer Mode's Model Retraining controls (Section 7.1 item 11,
FR-11.1-11.3, ADL-23). Every action here is a thin call into
``ApiClient``'s retraining methods run in the background (``workers.py``)
-- FR-11.2's "never silently activate" is enforced by the backend itself
(``confirm_deployment`` has no default for ``confirmed`` there either);
this panel does not add or relax that requirement, it just can't skip
the confirmation step even if it wanted to, since ``confirmed=True`` is
only ever sent from the explicit "Confirm deployment" button handler.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from aiperfopt.applog import get_logger
from aiperfopt.frontend.api_client import ApiClient, ApiError
from aiperfopt.frontend.workers import run_in_background

logger = get_logger(__name__)


class RetrainingPanel(QWidget):
    def __init__(self, api_client: ApiClient, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api_client = api_client
        self._pending_candidate: dict[str, Any] | None = None

        outer = QVBoxLayout(self)

        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Active model version:"))
        self._active_version_label = QLabel("--")
        status_row.addWidget(self._active_version_label)
        status_row.addStretch(1)
        self._refresh_active_button = QPushButton("Refresh")
        self._refresh_active_button.clicked.connect(self.refresh_active_version)
        status_row.addWidget(self._refresh_active_button)
        outer.addLayout(status_row)

        run_box = QGroupBox("Run retraining")
        run_layout = QHBoxLayout(run_box)
        run_layout.addWidget(QLabel("New version name:"))
        self._new_version_edit = QLineEdit()
        self._new_version_edit.setPlaceholderText("e.g. v2")
        run_layout.addWidget(self._new_version_edit)
        run_layout.addWidget(QLabel("Metrics (blank = all):"))
        self._target_metrics_edit = QLineEdit()
        self._target_metrics_edit.setPlaceholderText("cpu_percent, mem_percent")
        run_layout.addWidget(self._target_metrics_edit)
        self._run_button = QPushButton("Run retraining")
        self._run_button.clicked.connect(self._on_run_clicked)
        run_layout.addWidget(self._run_button)
        outer.addWidget(run_box)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        outer.addWidget(self._status_label)

        result_box = QGroupBox("Latest comparison")
        result_layout = QVBoxLayout(result_box)
        self._recommendation_label = QLabel("No candidate evaluated yet.")
        result_layout.addWidget(self._recommendation_label)
        self._reasons_list = QListWidget()
        result_layout.addWidget(self._reasons_list)

        result_layout.addWidget(QLabel("Quality gate (Phase 6 -- blocks deployment even if confirmed):"))
        self._gate_label = QLabel("--")
        result_layout.addWidget(self._gate_label)
        self._gate_criteria_list = QListWidget()
        result_layout.addWidget(self._gate_criteria_list)

        decision_row = QHBoxLayout()
        self._confirm_button = QPushButton("Confirm deployment")
        self._confirm_button.setEnabled(False)
        self._confirm_button.clicked.connect(lambda: self._confirm(True))
        self._reject_button = QPushButton("Reject candidate")
        self._reject_button.setEnabled(False)
        self._reject_button.clicked.connect(lambda: self._confirm(False))
        self._rollback_button = QPushButton("Rollback to last known good")
        self._rollback_button.clicked.connect(self._on_rollback_clicked)
        decision_row.addWidget(self._confirm_button)
        decision_row.addWidget(self._reject_button)
        decision_row.addWidget(self._rollback_button)
        result_layout.addLayout(decision_row)
        outer.addWidget(result_box)

        outer.addStretch(1)

    def refresh_active_version(self) -> None:
        run_in_background(
            self._api_client.get_active_model_version,
            on_success=lambda r: self._active_version_label.setText(str(r.get("active_model_version") or "--")),
            on_error=self._on_error,
        )

    def _on_run_clicked(self) -> None:
        new_version = self._new_version_edit.text().strip()
        if not new_version:
            self._status_label.setText("Enter a new model version name first.")
            return

        target_metrics_text = self._target_metrics_edit.text().strip()
        target_metrics = (
            [m.strip() for m in target_metrics_text.split(",") if m.strip()]
            if target_metrics_text
            else None
        )
        self._set_busy(True, "Running retraining -- this can take a while for large datasets...")

        def _call():
            return self._api_client.run_retraining(
                new_model_version=new_version, target_metrics=target_metrics
            )

        run_in_background(_call, on_success=self._on_candidate_ready, on_error=self._on_error)

    def _on_candidate_ready(self, candidate: dict[str, Any]) -> None:
        self._set_busy(False, f"Candidate {candidate['model_version']!r} evaluated.")
        self._pending_candidate = candidate
        comparison = candidate["comparison"]
        self._recommendation_label.setText(
            f"Recommendation: {comparison['recommendation']}  "
            f"(candidate {comparison['candidate_version']} vs. active {comparison['active_version']})"
        )
        self._reasons_list.clear()
        for reason in comparison.get("reasons") or []:
            self._reasons_list.addItem(reason)

        gate = candidate["gate"]
        self._gate_label.setText("PASS" if gate["passed"] else "FAIL -- deployment will be refused")
        self._gate_criteria_list.clear()
        for c in gate.get("forecast_criteria") or []:
            status = "pass" if c["passed"] else "FAIL"
            self._gate_criteria_list.addItem(
                f"[{status}] {c['metric_name']}/{c['horizon_label']}: mae={c['candidate_mae']} "
                f"(persistence={c['persistence_mae']}) directional={c['candidate_directional_accuracy']}"
                + (f" -- {c['note']}" if c.get("note") else "")
            )
        contention_criterion = gate.get("contention_criterion")
        if contention_criterion is not None:
            status = "pass" if contention_criterion["passed"] else "FAIL"
            self._gate_criteria_list.addItem(
                f"[{status}] contention: f1={contention_criterion['candidate_f1']} "
                f"(threshold-rule baseline={contention_criterion['baseline_f1']})"
                + (f" -- {contention_criterion['note']}" if contention_criterion.get("note") else "")
            )

        self._confirm_button.setEnabled(True)
        self._reject_button.setEnabled(True)

    def _confirm(self, confirmed: bool) -> None:
        if self._pending_candidate is None:
            return
        self._set_busy(True, "Confirming..." if confirmed else "Rejecting...")

        def _call():
            return self._api_client.confirm_deployment(
                candidate_version=self._pending_candidate["model_version"],
                comparison=self._pending_candidate["comparison"],
                gate=self._pending_candidate["gate"],
                confirmed=confirmed,
            )

        run_in_background(_call, on_success=self._on_confirmed, on_error=self._on_error)

    def _on_confirmed(self, deployment: dict[str, Any]) -> None:
        self._set_busy(False, f"Decision recorded: {deployment.get('decision')}")
        self._confirm_button.setEnabled(False)
        self._reject_button.setEnabled(False)
        self._pending_candidate = None
        self.refresh_active_version()

    def _on_rollback_clicked(self) -> None:
        self._set_busy(True, "Rolling back...")

        def _call():
            return self._api_client.rollback(reason="Manual rollback requested from Developer Mode.")

        run_in_background(_call, on_success=self._on_confirmed, on_error=self._on_error)

    def _set_busy(self, busy: bool, message: str) -> None:
        self._status_label.setText(message)
        self._run_button.setEnabled(not busy)
        self._rollback_button.setEnabled(not busy)

    def _on_error(self, exc: Exception) -> None:
        self._set_busy(False, "")
        detail = exc.detail if isinstance(exc, ApiError) else str(exc)
        self._status_label.setText(f"Failed: {detail}")
        logger.warning("retraining panel action failed", extra={"error": str(exc)})
