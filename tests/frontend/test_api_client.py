from __future__ import annotations

import time
from pathlib import Path

import pytest

from aiperfopt.frontend.api_client import ApiClient, ApiError
from aiperfopt.frontend.config import FrontendSettings
from aiperfopt.storage import insert_system_metric_samples_batch
from tests.api_conftest import build_test_ctx, rising_cpu_samples_now
from tests.frontend.server_fixture import running_backend


def _client_for(base_url: str) -> ApiClient:
    host, port = base_url.replace("http://", "").split(":")
    return ApiClient(settings=FrontendSettings(api_host=host, api_port=int(port)))


def test_health_and_config(tmp_path: Path):
    with running_backend(tmp_path) as (base_url, _ctx):
        with _client_for(base_url) as client:
            assert client.health() == {"status": "ok"}
            config = client.get_config()
            assert "mode" in config
            assert config["mode"]["active_mode"] == "general"


def test_monitoring_and_live_start_stop(tmp_path: Path):
    with running_backend(tmp_path) as (base_url, _ctx):
        with _client_for(base_url) as client:
            assert client.monitoring_status() == {"running": False}
            assert client.start_monitoring() == {"running": True}
            assert client.monitoring_status() == {"running": True}
            assert client.stop_monitoring() == {"running": False}

            assert client.live_status() == {"running": False}
            assert client.start_live_inference() == {"running": True}
            assert client.stop_live_inference() == {"running": False}


def test_live_state_is_none_before_loop_ever_runs(tmp_path: Path):
    with running_backend(tmp_path) as (base_url, _ctx):
        with _client_for(base_url) as client:
            assert client.get_live_state() is None


def test_history_endpoints_return_inserted_data(tmp_path: Path):
    ctx = build_test_ctx(tmp_path)
    samples = rising_cpu_samples_now(n=20, interval=2.0)
    insert_system_metric_samples_batch(ctx.storage_engine, samples)

    with running_backend(tmp_path, ctx=ctx) as (base_url, _ctx):
        with _client_for(base_url) as client:
            rows = client.get_system_metrics(limit=5)
            assert len(rows) == 5
            assert "cpu_percent" in rows[0]


def test_unknown_route_raises_api_error(tmp_path: Path):
    with running_backend(tmp_path) as (base_url, _ctx):
        with _client_for(base_url) as client:
            with pytest.raises(ApiError) as exc_info:
                client._get("/definitely/not/a/real/route")
            assert exc_info.value.status_code == 404


def test_predicted_vs_actual_rejects_unknown_metric(tmp_path: Path):
    with running_backend(tmp_path) as (base_url, _ctx):
        with _client_for(base_url) as client:
            with pytest.raises(ApiError) as exc_info:
                client.get_predicted_vs_actual("not_a_metric")
            assert exc_info.value.status_code == 400


def test_reportable_metric_names(tmp_path: Path):
    with running_backend(tmp_path) as (base_url, _ctx):
        with _client_for(base_url) as client:
            names = client.get_reportable_metric_names()
            assert "cpu_percent" in names


def test_report_summary_with_period(tmp_path: Path):
    with running_backend(tmp_path) as (base_url, _ctx):
        with _client_for(base_url) as client:
            report = client.get_report_summary(period="daily")
            assert "system_metrics" in report
            assert "diagnosis" in report


def test_report_summary_without_period_or_window_is_400(tmp_path: Path):
    with running_backend(tmp_path) as (base_url, _ctx):
        with _client_for(base_url) as client:
            with pytest.raises(ApiError) as exc_info:
                client.get_report_summary()
            assert exc_info.value.status_code == 400


def test_retraining_round_trip(tmp_path: Path):
    ctx = build_test_ctx(tmp_path)
    samples = rising_cpu_samples_now(n=300, interval=2.0)
    insert_system_metric_samples_batch(ctx.storage_engine, samples)

    with running_backend(tmp_path, ctx=ctx) as (base_url, _ctx):
        with _client_for(base_url) as client:
            assert client.get_active_model_version()["active_model_version"] is None

            candidate = client.run_retraining(
                new_model_version="v1", data_source="real_usage", target_metrics=["cpu_percent"],
            )
            assert candidate["model_version"] == "v1"

            deployment = client.confirm_deployment(
                candidate_version="v1", comparison=candidate["comparison"], gate=candidate["gate"], confirmed=True,
            )
            assert deployment["decision"] == "deployed"
            assert client.get_active_model_version()["active_model_version"] == "v1"
