from __future__ import annotations

import pytest

from aiperfopt.frontend.widgets.dashboard import DashboardWidget

_BASE_STATE = {
    "timestamp": 1000.0,
    "latest_metrics": {"cpu_percent": 42.5, "mem_percent": 61.2},
    "feature_vector": None,
    "active_model_version": None,
    "contention": None,
    "diagnosis": None,
    "recommendation": None,
    "anomaly_notification": False,
    "error": None,
}


@pytest.mark.behavioural
def test_renders_current_metrics(qtbot):
    widget = DashboardWidget()
    qtbot.addWidget(widget)
    widget.update_state(_BASE_STATE)
    assert widget._cpu_tile._value_label.text() == "42.5%"
    assert widget._mem_tile._value_label.text() == "61.2%"


@pytest.mark.behavioural
def test_anomaly_banner_toggles(qtbot):
    widget = DashboardWidget()
    qtbot.addWidget(widget)

    widget.update_state(_BASE_STATE)
    assert widget._anomaly_banner.isHidden()

    state = dict(_BASE_STATE, anomaly_notification=True)
    widget.update_state(state)
    assert not widget._anomaly_banner.isHidden()
    assert "Unusual" in widget._anomaly_banner.text()


@pytest.mark.behavioural
def test_diagnosis_renders_plain_language_text(qtbot):
    widget = DashboardWidget()
    qtbot.addWidget(widget)

    state = dict(
        _BASE_STATE,
        diagnosis={
            "category": "cpu_bound_single_process",
            "plain_language_explanation": "One app is hogging your CPU right now.",
            "confidence": 0.82,
            "is_actionable": True,
        },
    )
    widget.update_state(state)
    assert "One process is using most of your CPU" == widget._diagnosis_title.text()
    assert "hogging" in widget._diagnosis_body.text()
    assert "82%" in widget._diagnosis_confidence.text()


@pytest.mark.plumbing
def test_no_diagnosis_shows_default_text(qtbot):
    widget = DashboardWidget()
    qtbot.addWidget(widget)
    widget.update_state(_BASE_STATE)
    assert widget._diagnosis_title.text() == "No issue detected right now"


@pytest.mark.plumbing
def test_recommendation_actions_render_as_cards(qtbot):
    widget = DashboardWidget()
    qtbot.addWidget(widget)

    state = dict(
        _BASE_STATE,
        recommendation={
            "diagnosis_id": 1,
            "diagnosis_category": "cpu_bound_single_process",
            "actions": [
                {"title": "Quit the process", "description": "Close it.", "rationale": "It's using too much CPU."},
                {"title": "Lower priority", "description": "Renice it.", "rationale": "Reduces impact."},
            ],
            "catalog_version": "v1",
        },
    )
    widget.update_state(state)
    assert widget._recs_layout.count() == 2
    assert widget._recs_empty_label.isHidden()


@pytest.mark.plumbing
def test_no_recommendation_shows_empty_state(qtbot):
    widget = DashboardWidget()
    qtbot.addWidget(widget)
    widget.update_state(_BASE_STATE)
    assert not widget._recs_empty_label.isHidden()


@pytest.mark.behavioural
def test_error_state_shows_error_banner(qtbot):
    widget = DashboardWidget()
    qtbot.addWidget(widget)
    state = dict(_BASE_STATE, error="storage unavailable")
    widget.update_state(state)
    assert "storage unavailable" in widget._status_banner.text()


@pytest.mark.behavioural
def test_readiness_drives_the_banner_text_and_color(qtbot):
    widget = DashboardWidget()
    qtbot.addWidget(widget)

    widget.update_readiness({
        "state": "collecting", "detail": "Collecting - 30s of 300s needed.",
        "history_seconds": 30.0, "history_seconds_required": 300.0, "can_predict": False,
    })
    assert "Collecting" in widget._status_banner.text()

    widget.update_readiness({
        "state": "ready", "detail": "Ready.", "history_seconds": 300.0,
        "history_seconds_required": 300.0, "can_predict": True,
    })
    assert widget._status_banner.text() == "Live"


@pytest.mark.behavioural
def test_replay_readiness_state_shows_unmistakable_replay_banner(qtbot):
    widget = DashboardWidget()
    qtbot.addWidget(widget)

    widget.update_readiness({
        "state": "replay", "detail": "REPLAY - Ready.", "history_seconds": 300.0,
        "history_seconds_required": 300.0, "can_predict": True,
    })
    assert widget._status_banner.text() == "REPLAY"


@pytest.mark.behavioural
def test_live_state_error_overrides_readiness_banner(qtbot):
    widget = DashboardWidget()
    qtbot.addWidget(widget)

    widget.update_readiness({
        "state": "ready", "detail": "Ready.", "history_seconds": 300.0,
        "history_seconds_required": 300.0, "can_predict": True,
    })
    widget.update_state(dict(_BASE_STATE, error="storage unavailable"))
    assert "storage unavailable" in widget._status_banner.text()

    # Clearing the error falls back to the last-known readiness, not "Live".
    widget.update_state(_BASE_STATE)
    assert widget._status_banner.text() == "Live"


@pytest.mark.plumbing
def test_recommendation_cards_are_cleared_between_updates(qtbot):
    widget = DashboardWidget()
    qtbot.addWidget(widget)

    state_with_actions = dict(
        _BASE_STATE,
        recommendation={
            "diagnosis_id": 1, "diagnosis_category": "x",
            "actions": [{"title": "A", "description": "B", "rationale": "C"}],
            "catalog_version": "v1",
        },
    )
    widget.update_state(state_with_actions)
    assert widget._recs_layout.count() == 1

    widget.update_state(_BASE_STATE)
    assert not widget._recs_empty_label.isHidden()
