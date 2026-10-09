from __future__ import annotations

from pathlib import Path

import pytest

from tests.api_conftest import build_test_client


@pytest.mark.plumbing
def test_active_version_none_initially(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        r = client.get("/retraining/active-version")
        assert r.status_code == 200
        assert r.json() == {"active_model_version": None}


@pytest.mark.plumbing
def test_run_retraining_real_usage_requires_data(tmp_path: Path):
    with build_test_client(tmp_path, with_history=False) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "data_source": "real_usage",
        })
        assert r.status_code == 422  # RetrainingError -> not enough data


@pytest.mark.plumbing
def test_run_retraining_real_usage_success(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "target_metrics": ["cpu_percent"],
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model_version"] == "v1"
        assert body["data_source"] == "real_usage"
        assert body["comparison"]["recommendation"] == "no_active_baseline"


@pytest.mark.plumbing
def test_confirm_deployment_activates_version(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "target_metrics": ["cpu_percent"],
        })
        candidate = r.json()

        r = client.post("/retraining/confirm", json={
            "candidate_version": candidate["model_version"], "comparison": candidate["comparison"],
            "gate": candidate["gate"],
            "confirmed": True, "confirmed_by": "test-suite",
        })
        assert r.status_code == 200
        assert r.json()["decision"] == "deployed"

        r = client.get("/retraining/active-version")
        assert r.json() == {"active_model_version": "v1"}


@pytest.mark.behavioural
def test_confirm_deployment_on_a_failing_gate_returns_422_with_reasons(tmp_path: Path):
    """The router maps a gate failure to 422 listing the failed criteria
    (Phase 6, point 2) -- the confirm endpoint is stateless and trusts
    the client to round-trip exactly what ``/retraining/run`` showed it
    (see ``_gate_from_request``'s docstring), so a failing gate is
    exercised the same way: substitute a failing one for what a real
    candidate would otherwise pass."""
    with build_test_client(tmp_path, with_history=True) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "target_metrics": ["cpu_percent"],
        })
        candidate = r.json()
        failing_gate = {
            "passed": False,
            "reasons": ["contention_classifier: F1 0.100 falls short of threshold-rule baseline 0.500 by more than the allowed 0.050"],
            "forecast_criteria": [],
            "contention_criterion": None,
        }

        r = client.post("/retraining/confirm", json={
            "candidate_version": candidate["model_version"], "comparison": candidate["comparison"],
            "gate": failing_gate, "confirmed": True,
        })
        assert r.status_code == 422
        assert "contention_classifier" in r.json()["detail"]

        r = client.get("/retraining/active-version")
        assert r.json() == {"active_model_version": None}


@pytest.mark.behavioural
def test_confirm_deployment_rejected_does_not_activate(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "target_metrics": ["cpu_percent"],
        })
        candidate = r.json()

        r = client.post("/retraining/confirm", json={
            "candidate_version": candidate["model_version"], "comparison": candidate["comparison"],
            "gate": candidate["gate"],
            "confirmed": False,
        })
        assert r.status_code == 200
        assert r.json()["decision"] == "rejected"

        r = client.get("/retraining/active-version")
        assert r.json() == {"active_model_version": None}


@pytest.mark.plumbing
def test_rollback_with_no_prior_version_returns_409(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "target_metrics": ["cpu_percent"],
        })
        candidate = r.json()
        client.post("/retraining/confirm", json={
            "candidate_version": candidate["model_version"], "comparison": candidate["comparison"],
            "gate": candidate["gate"],
            "confirmed": True,
        })

        r = client.post("/retraining/rollback", json={"reason": "test"})
        assert r.status_code == 409


@pytest.mark.behavioural
def test_rollback_restores_previous_version(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True) as client:
        r = client.post("/retraining/run", json={
            "new_model_version": "v1", "target_metrics": ["cpu_percent"],
        })
        candidate1 = r.json()
        client.post("/retraining/confirm", json={
            "candidate_version": candidate1["model_version"], "comparison": candidate1["comparison"],
            "gate": candidate1["gate"],
            "confirmed": True,
        })

        r = client.post("/retraining/run", json={
            "new_model_version": "v2", "target_metrics": ["cpu_percent"],
        })
        candidate2 = r.json()
        client.post("/retraining/confirm", json={
            "candidate_version": candidate2["model_version"], "comparison": candidate2["comparison"],
            "gate": candidate2["gate"],
            "confirmed": True,
        })
        assert client.get("/retraining/active-version").json()["active_model_version"] == "v2"

        r = client.post("/retraining/rollback", json={"reason": "v2 failed at inference"})
        assert r.status_code == 200
        assert r.json()["candidate_version"] == "v1"
        assert client.get("/retraining/active-version").json()["active_model_version"] == "v1"
