from __future__ import annotations

from pathlib import Path

from tests.api_conftest import build_test_client


def test_health(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_config_returns_active_settings(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        r = client.get("/config")
        assert r.status_code == 200
        body = r.json()
        assert "forecasting" in body
        assert "retraining" in body
        assert "api" in body


def test_monitoring_never_running_by_default(tmp_path: Path):
    with build_test_client(tmp_path, auto_start=False) as client:
        r = client.get("/monitoring/status")
        assert r.status_code == 200
        assert r.json() == {"running": False}


def test_monitoring_start_and_stop(tmp_path: Path):
    with build_test_client(tmp_path, auto_start=False) as client:
        r = client.post("/monitoring/start")
        assert r.status_code == 200
        assert r.json() == {"running": True}

        r = client.get("/monitoring/status")
        assert r.json() == {"running": True}

        r = client.post("/monitoring/stop")
        assert r.status_code == 200
        assert r.json() == {"running": False}


def test_live_never_running_by_default(tmp_path: Path):
    with build_test_client(tmp_path, auto_start=False) as client:
        r = client.get("/live/status")
        assert r.status_code == 200
        assert r.json() == {"running": False}


def test_live_start_and_stop(tmp_path: Path):
    with build_test_client(tmp_path, auto_start=False) as client:
        r = client.post("/live/start")
        assert r.status_code == 200
        assert r.json() == {"running": True}

        r = client.post("/live/stop")
        assert r.status_code == 200
        assert r.json() == {"running": False}


# --- readiness (Phase 0) -------------------------------------------------
#
# The live loop returns early on two conditions and previously did so
# silently while the UI showed a green "Live" banner. These assert the
# endpoint the UI uses instead of that hardcoded banner.


def test_readiness_reports_no_data_on_a_fresh_database(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        r = client.get("/readiness")
        assert r.status_code == 200

        body = r.json()
        assert body["state"] == "no_data"
        assert body["can_predict"] is False
        assert body["active_model_version"] is None
        assert body["history_seconds"] == 0.0
        assert body["history_seconds_required"] > 0


def test_readiness_reports_needs_training_once_history_exists(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True) as client:
        body = client.get("/readiness").json()

        assert body["state"] == "needs_training"
        assert body["can_predict"] is False
        assert body["history_seconds"] >= body["history_seconds_required"]
        assert "aiperfopt train" in body["detail"]
