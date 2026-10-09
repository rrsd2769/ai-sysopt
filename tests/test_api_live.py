from __future__ import annotations

import math
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aiperfopt.api.app import create_app
from aiperfopt.contention.persistence import contention_artifact_exists
from aiperfopt.storage import insert_system_metric_samples_batch
from aiperfopt.storage.reader import get_forecast_predictions_range
from aiperfopt.storage.records import SystemMetricSample
from tests.api_conftest import build_test_client, build_test_ctx


@pytest.mark.plumbing
def test_live_state_is_none_before_loop_starts(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True) as client:
        r = client.get("/live/state")
        assert r.status_code == 200
        assert r.json() is None


@pytest.mark.plumbing
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


@pytest.mark.plumbing
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


@pytest.mark.plumbing
def test_live_loop_produces_real_prediction_after_deployment(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True, auto_start=False) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "target_metrics": ["cpu_percent"],
        })
        candidate = r.json()
        client.post("/retraining/confirm", json={
            "candidate_version": candidate["model_version"], "comparison": candidate["comparison"],
            "gate": candidate["gate"],
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


@pytest.mark.plumbing
def test_websocket_receives_broadcast_state(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True, auto_start=False) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "target_metrics": ["cpu_percent"],
        })
        candidate = r.json()
        client.post("/retraining/confirm", json={
            "candidate_version": candidate["model_version"], "comparison": candidate["comparison"],
            "gate": candidate["gate"],
            "confirmed": True,
        })
        client.post("/live/start")
        time.sleep(2.5)

        with client.websocket_connect("/ws/live") as ws:
            msg = ws.receive_json()
            assert msg["active_model_version"] == "v1"
            assert "timestamp" in msg

        client.post("/live/stop")


@pytest.mark.plumbing
def test_websocket_connects_before_any_state_exists(tmp_path: Path):
    """Connecting before the loop has ever ticked must not hang or error --
    there's simply nothing to send yet."""
    with build_test_client(tmp_path, with_history=True, auto_start=False) as client:
        with client.websocket_connect("/ws/live") as ws:
            pass  # connecting and disconnecting cleanly is the assertion


@pytest.mark.plumbing
def test_live_tick_persists_one_forecast_row_per_deployed_metric(tmp_path: Path):
    """Phase 7 / F5: a live tick with a deployed forecaster must write a
    ForecastPredictionRecord, not just compute one and discard it."""
    with build_test_client(tmp_path, with_history=True, auto_start=False) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "target_metrics": ["cpu_percent"],
        })
        candidate = r.json()
        client.post("/retraining/confirm", json={
            "candidate_version": candidate["model_version"], "comparison": candidate["comparison"],
            "gate": candidate["gate"],
            "confirmed": True,
        })

        client.post("/live/start")
        time.sleep(2.5)
        client.post("/live/stop")

        state = client.get("/live/state").json()
        assert state["forecasts"]
        forecast = next(f for f in state["forecasts"] if f["metric_name"] == "cpu_percent")
        assert forecast["model_version"] == "v1"
        assert forecast["target_timestamp"] == pytest.approx(
            forecast["generated_at"] + forecast["horizon_seconds"]
        )

        engine = client.app.state.ctx.storage_engine
        rows = get_forecast_predictions_range(engine, metric_name="cpu_percent")
        assert len(rows) >= 1
        assert rows[0].target_timestamp == pytest.approx(rows[0].generated_at + rows[0].horizon_seconds)


@pytest.mark.plumbing
def test_persisted_forecasts_feed_predicted_vs_actual_and_summary(tmp_path: Path):
    """After ticks plus a later real sample landing near the forecast's
    target, /reporting/predicted-vs-actual pairs it up and
    /reporting/summary's prediction_accuracy is populated."""
    with build_test_client(tmp_path, with_history=True, auto_start=False) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "target_metrics": ["cpu_percent"],
        })
        candidate = r.json()
        client.post("/retraining/confirm", json={
            "candidate_version": candidate["model_version"], "comparison": candidate["comparison"],
            "gate": candidate["gate"],
            "confirmed": True,
        })

        client.post("/live/start")
        time.sleep(2.5)
        client.post("/live/stop")

        state = client.get("/live/state").json()
        forecast = next(f for f in state["forecasts"] if f["metric_name"] == "cpu_percent")

        engine = client.app.state.ctx.storage_engine
        insert_system_metric_samples_batch(engine, [
            SystemMetricSample(timestamp=forecast["target_timestamp"], cpu_percent=42.0, mem_percent=50.0)
        ])

        pva = client.get("/reporting/predicted-vs-actual", params={"metric_name": "cpu_percent"})
        assert pva.status_code == 200
        points = pva.json()
        assert points
        assert any(p["actual_value"] is not None for p in points)

        summary = client.get("/reporting/summary", params={
            "start": forecast["generated_at"] - 60, "end": forecast["target_timestamp"] + 60,
        })
        assert summary.status_code == 200
        assert summary.json()["prediction_accuracy"]


def _flat_low_cpu_samples_now(n: int = 300, interval: float = 2.0) -> list[SystemMetricSample]:
    """Real wall-clock timestamps, like ``rising_cpu_samples_now``, but
    capped well below the 70% contention labeling threshold -- produces
    single-class (all-negative) contention training data on purpose."""
    now = time.time()
    return [
        SystemMetricSample(
            timestamp=now - (n - i) * interval,
            cpu_percent=max(0.0, min(40.0, 5.0 + (i / n) * 20.0 + 4.0 * math.sin(i / 3.0))),
            mem_percent=50.0,
        )
        for i in range(n)
    ]


@pytest.mark.behavioural
def test_live_tick_persists_forecasts_when_no_contention_classifier_trained(tmp_path: Path):
    """Real, expected scenario per Phase 5/6: accumulated data can be
    single-class for contention, so no classifier is ever trained/saved
    for that model version. Phase 7's forecast persistence must not be
    held hostage to that -- only contention/diagnosis/recommendation
    should degrade for that cycle (NFR-11.1), and this must not be
    mistaken for a corrupted deployment needing rollback."""
    ctx = build_test_ctx(tmp_path)
    insert_system_metric_samples_batch(ctx.storage_engine, _flat_low_cpu_samples_now())
    app = create_app(ctx, auto_start=False)
    with TestClient(app) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "target_metrics": ["cpu_percent"],
        })
        candidate = r.json()
        client.post("/retraining/confirm", json={
            "candidate_version": candidate["model_version"], "comparison": candidate["comparison"],
            "gate": candidate["gate"],
            "confirmed": True,
        })

        assert not contention_artifact_exists(ctx.settings.forecasting.models_dir, "v1")

        client.post("/live/start")
        time.sleep(2.5)

        state = client.get("/live/state").json()
        assert state["active_model_version"] == "v1"
        assert state["error"] is None
        assert state["forecasts"]
        assert state["contention"] is None
        client.post("/live/stop")
