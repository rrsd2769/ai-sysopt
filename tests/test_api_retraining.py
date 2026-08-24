from __future__ import annotations

from pathlib import Path

from tests.api_conftest import build_test_client


def test_active_version_none_initially(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        r = client.get("/retraining/active-version")
        assert r.status_code == 200
        assert r.json() == {"active_model_version": None}


def test_run_retraining_real_usage_requires_data(tmp_path: Path):
    with build_test_client(tmp_path, with_history=False) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "data_source": "real_usage",
        })
        assert r.status_code == 422  # RetrainingError -> not enough data


def test_run_retraining_real_usage_success(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "data_source": "real_usage", "target_metrics": ["cpu_percent"],
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model_version"] == "v1"
        assert body["data_source"] == "real_usage"
        assert body["comparison"]["recommendation"] == "no_active_baseline"


def test_run_retraining_synthetic_success(tmp_path: Path):
    with build_test_client(tmp_path, with_history=False) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "data_source": "synthetic", "target_metrics": ["cpu_percent"],
            "scenario_duration_seconds": 150, "interval_seconds": 2.0,
        })
        assert r.status_code == 200, r.text
        assert r.json()["data_source"] == "synthetic"


def test_confirm_deployment_activates_version(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "data_source": "real_usage", "target_metrics": ["cpu_percent"],
        })
        candidate = r.json()

        r = client.post("/retraining/confirm", json={
            "candidate_version": candidate["model_version"], "comparison": candidate["comparison"],
            "confirmed": True, "confirmed_by": "test-suite",
        })
        assert r.status_code == 200
        assert r.json()["decision"] == "deployed"

        r = client.get("/retraining/active-version")
        assert r.json() == {"active_model_version": "v1"}


def test_confirm_deployment_rejected_does_not_activate(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "data_source": "real_usage", "target_metrics": ["cpu_percent"],
        })
        candidate = r.json()

        r = client.post("/retraining/confirm", json={
            "candidate_version": candidate["model_version"], "comparison": candidate["comparison"],
            "confirmed": False,
        })
        assert r.status_code == 200
        assert r.json()["decision"] == "rejected"

        r = client.get("/retraining/active-version")
        assert r.json() == {"active_model_version": None}


def test_rollback_with_no_prior_version_returns_409(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "data_source": "real_usage", "target_metrics": ["cpu_percent"],
        })
        candidate = r.json()
        client.post("/retraining/confirm", json={
            "candidate_version": candidate["model_version"], "comparison": candidate["comparison"],
            "confirmed": True,
        })

        r = client.post("/retraining/rollback", json={"reason": "test"})
        assert r.status_code == 409


def test_rollback_restores_previous_version(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "data_source": "real_usage", "target_metrics": ["cpu_percent"],
        })
        candidate1 = r.json()
        client.post("/retraining/confirm", json={
            "candidate_version": candidate1["model_version"], "comparison": candidate1["comparison"],
            "confirmed": True,
        })

        r = client.post("/retraining/run", json={
            "new_model_version": "v2", "data_source": "real_usage", "target_metrics": ["cpu_percent"],
        })
        candidate2 = r.json()
        client.post("/retraining/confirm", json={
            "candidate_version": candidate2["model_version"], "comparison": candidate2["comparison"],
            "confirmed": True,
        })
        assert client.get("/retraining/active-version").json()["active_model_version"] == "v2"

        r = client.post("/retraining/rollback", json={"reason": "v2 failed at inference"})
        assert r.status_code == 200
        assert r.json()["candidate_version"] == "v1"
        assert client.get("/retraining/active-version").json()["active_model_version"] == "v1"
