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
