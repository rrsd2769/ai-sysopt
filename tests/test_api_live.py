from __future__ import annotations

import time
from pathlib import Path

from tests.api_conftest import build_test_client


def test_live_state_is_none_before_loop_starts(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True) as client:
        r = client.get("/live/state")
        assert r.status_code == 200
        assert r.json() is None


def test_live_state_degrades_gracefully_with_no_history(tmp_path: Path):
    """No history at all -> the loop should tick without error and report nothing yet."""
    with build_test_client(tmp_path, with_history=False, auto_start=False) as client:
        client.post("/live/start")
        time.sleep(1.5)
        r = client.get("/live/state")
        state = r.json()
        assert state is not None
        assert state["feature_vector"] is None
        assert state["active_model_version"] is None
        assert state["error"] is None
        client.post("/live/stop")


def test_live_state_degrades_gracefully_with_no_deployed_model(tmp_path: Path):
    """History exists (features computable) but nothing was ever deployed."""
    with build_test_client(tmp_path, with_history=True, auto_start=False) as client:
        client.post("/live/start")
        time.sleep(1.5)
        r = client.get("/live/state")
        state = r.json()
        assert state is not None
        assert state["feature_vector"] is not None
        assert state["active_model_version"] is None
        assert state["contention"] is None
        client.post("/live/stop")


def test_live_loop_produces_real_prediction_after_deployment(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True, auto_start=False) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "data_source": "real_usage", "target_metrics": ["cpu_percent"],
        })
        candidate = r.json()
        client.post("/retraining/confirm", json={
            "candidate_version": candidate["model_version"], "comparison": candidate["comparison"],
            "confirmed": True,
        })

        client.post("/live/start")
        time.sleep(2.5)

        r = client.get("/live/state")
        state = r.json()
        assert state["active_model_version"] == "v1"
        assert state["contention"] is not None
        assert 0.0 <= state["contention"]["probability"] <= 1.0
        client.post("/live/stop")


def test_websocket_receives_broadcast_state(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True, auto_start=False) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "data_source": "real_usage", "target_metrics": ["cpu_percent"],
        })
        candidate = r.json()
        client.post("/retraining/confirm", json={
            "candidate_version": candidate["model_version"], "comparison": candidate["comparison"],
            "confirmed": True,
        })
        client.post("/live/start")
        time.sleep(2.5)

        with client.websocket_connect("/ws/live") as ws:
            msg = ws.receive_json()
            assert msg["active_model_version"] == "v1"
            assert "timestamp" in msg

        client.post("/live/stop")


def test_websocket_connects_before_any_state_exists(tmp_path: Path):
    """Connecting before the loop has ever ticked must not hang or error --
    there's simply nothing to send yet."""
    with build_test_client(tmp_path, with_history=True, auto_start=False) as client:
        with client.websocket_connect("/ws/live") as ws:
            pass  # connecting and disconnecting cleanly is the assertion
