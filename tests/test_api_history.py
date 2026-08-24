from __future__ import annotations

from pathlib import Path

from aiperfopt.storage import insert_evaluation, insert_model_deployment
from aiperfopt.storage.records import EvaluationRecord, ModelDeploymentRecord
from tests.api_conftest import build_test_client


def test_system_metrics_empty_database(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        r = client.get("/metrics/system")
        assert r.status_code == 200
        assert r.json() == []


def test_system_metrics_returns_inserted_rows(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True) as client:
        r = client.get("/metrics/system")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 300
        assert "cpu_percent" in body[0]


def test_system_metrics_respects_limit(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True) as client:
        r = client.get("/metrics/system", params={"limit": 10})
        assert r.status_code == 200
        assert len(r.json()) == 10


def test_evaluations_by_version_endpoint(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        engine = client.app.state.ctx.storage_engine
        insert_evaluation(
            engine,
            EvaluationRecord(
                evaluated_at=100.0, model_name="lightgbm", model_version="v1",
                metric_name="cpu_percent", horizon_label="short", dataset_tag="holdout_real_usage",
                mae=1.0, rmse=2.0, directional_accuracy=0.8,
            ),
        )
        r = client.get("/evaluations/by-version/v1")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 1
        assert body[0]["model_version"] == "v1"

        r = client.get("/evaluations/by-version/no-such-version")
        assert r.status_code == 200
        assert r.json() == []


def test_deployments_endpoints(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        engine = client.app.state.ctx.storage_engine
        insert_model_deployment(
            engine,
            ModelDeploymentRecord(
                decided_at=100.0, candidate_version="v1", decision="deployed",
                recommendation="no_active_baseline", confirmed_by="human",
            ),
        )
        r = client.get("/deployments")
        assert r.status_code == 200
        assert len(r.json()) == 1

        r = client.get("/deployments/latest")
        assert r.status_code == 200
        assert r.json()["candidate_version"] == "v1"


def test_deployments_latest_none_when_empty(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        r = client.get("/deployments/latest")
        assert r.status_code == 200
        assert r.json() is None


def test_forecasts_contention_diagnoses_recommendations_empty_ok(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        for path in ("/forecasts", "/contention", "/diagnoses", "/recommendations", "/features", "/evaluations"):
            r = client.get(path)
            assert r.status_code == 200, path
            assert r.json() == []
