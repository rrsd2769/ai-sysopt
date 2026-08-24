from __future__ import annotations

from aiperfopt.frontend.widgets.developer_panel import DeveloperPanel

_BASE_STATE = {
    "timestamp": 1000.0,
    "latest_metrics": {"cpu_percent": 42.5},
    "feature_vector": {"cpu_percent_lag_1": 40.1, "cpu_percent_roll_mean_10s": 39.2},
    "active_model_version": "v3",
    "contention": None,
    "diagnosis": None,
    "recommendation": None,
    "anomaly_notification": False,
    "error": None,
}


def test_shows_active_model_version(qtbot):
    panel = DeveloperPanel()
    qtbot.addWidget(panel)
    panel.update_state(_BASE_STATE)
    assert "v3" in panel._model_version_label.text()


def test_feature_vector_populates_tree(qtbot):
    panel = DeveloperPanel()
    qtbot.addWidget(panel)
    panel.update_state(_BASE_STATE)
    assert panel._feature_vector_tree.tree.topLevelItemCount() == 2


def test_contention_contributing_metrics_and_processes(qtbot):
    panel = DeveloperPanel()
    qtbot.addWidget(panel)
    state = dict(
        _BASE_STATE,
        contention={
            "probability": 0.73, "threshold_used": 0.6, "triggered": True, "model_version": "v3",
            "contributing_metrics": [{"metric_name": "cpu_percent", "contribution": 0.55}],
            "contributing_processes": [{"pid": 123, "name": "chrome", "value": 88.1}],
            "anomaly_score": 0.1,
        },
    )
    panel.update_state(state)
    assert panel._contributing_metrics_tree.topLevelItemCount() == 1
    assert panel._contributing_processes_tree.topLevelItemCount() == 1
    assert "73.0%" in panel._contention_summary.text()


def test_no_contention_shows_placeholder(qtbot):
    panel = DeveloperPanel()
    qtbot.addWidget(panel)
    panel.update_state(_BASE_STATE)
    assert "No contention prediction" in panel._contention_summary.text()


def test_diagnosis_forecast_explanations_populate_tree(qtbot):
    panel = DeveloperPanel()
    qtbot.addWidget(panel)
    state = dict(
        _BASE_STATE,
        diagnosis={
            "category": "cpu_bound_single_process",
            "technical_explanation": "cpu_percent forecast rose sharply, dominant SHAP feature: cpu_percent_lag_1",
            "is_actionable": True,
            "confidence": 0.9,
            "explanation_tier": "shap",
            "trigger_reason": "high_confidence_contention",
            "forecast_explanations": [
                {
                    "metric_name": "cpu_percent", "forecaster_kind": "lightgbm", "tier": "shap",
                    "top_features": [{"feature_name": "cpu_percent_lag_1", "value": 0.42}],
                    "note": None,
                }
            ],
        },
    )
    panel.update_state(state)
    assert panel._forecast_explanations_tree.topLevelItemCount() == 1
    assert "cpu_percent forecast rose" in panel._diagnosis_technical.text()
    assert "shap" in panel._diagnosis_meta.text()


def test_trees_clear_between_updates(qtbot):
    panel = DeveloperPanel()
    qtbot.addWidget(panel)
    state_with_contention = dict(
        _BASE_STATE,
        contention={
            "probability": 0.73, "threshold_used": 0.6, "triggered": True, "model_version": "v3",
            "contributing_metrics": [{"metric_name": "cpu_percent", "contribution": 0.55}],
            "contributing_processes": [],
            "anomaly_score": 0.1,
        },
    )
    panel.update_state(state_with_contention)
    assert panel._contributing_metrics_tree.topLevelItemCount() == 1

    panel.update_state(_BASE_STATE)
    assert panel._contributing_metrics_tree.topLevelItemCount() == 0
