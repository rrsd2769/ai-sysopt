from __future__ import annotations

import time
from pathlib import Path

from tests.api_conftest import build_test_client, rising_cpu_samples_now

from aiperfopt.storage import insert_forecast_prediction
from aiperfopt.storage.records import ForecastPredictionRecord


def _insert_forecast(ctx, *, target_timestamp: float, predicted_value: float, horizon_label: str = "short"):
    insert_forecast_prediction(
        ctx.storage_engine,
        ForecastPredictionRecord(
            generated_at=target_timestamp - 30.0,
            target_timestamp=target_timestamp,
            metric_name="cpu_percent",
            horizon_label=horizon_label,
            horizon_seconds=30,
            predicted_value=predicted_value,
            model_version="v1",
            confidence=0.8,
        ),
    )


def test_rejects_unknown_metric_name(tmp_path: Path):
    with build_test_client(tmp_path, with_history=False) as client:
        r = client.get("/reporting/predicted-vs-actual", params={"metric_name": "not_a_real_metric"})
        assert r.status_code == 400


def test_empty_when_no_predictions_exist(tmp_path: Path):
    with build_test_client(tmp_path, with_history=True) as client:
        r = client.get("/reporting/predicted-vs-actual", params={"metric_name": "cpu_percent"})
        assert r.status_code == 200
        assert r.json() == []


def test_joins_prediction_with_nearest_actual_sample(tmp_path: Path):
    from tests.api_conftest import build_test_ctx

    ctx = build_test_ctx(tmp_path)
    samples = rising_cpu_samples_now(n=50, interval=2.0)
    from aiperfopt.storage import insert_system_metric_samples_batch

    insert_system_metric_samples_batch(ctx.storage_engine, samples)

    # Target a timestamp that lands very close to one of the real samples.
    target = samples[25].timestamp + 0.3
    _insert_forecast(ctx, target_timestamp=target, predicted_value=samples[25].cpu_percent + 5.0)

    from aiperfopt.api.app import create_app
    from fastapi.testclient import TestClient

    with TestClient(create_app(ctx)) as client:
        r = client.get("/reporting/predicted-vs-actual", params={"metric_name": "cpu_percent"})
        assert r.status_code == 200
        points = r.json()
        assert len(points) == 1
        point = points[0]
        assert point["actual_value"] == samples[25].cpu_percent
        assert point["predicted_value"] == samples[25].cpu_percent + 5.0
        assert abs(point["error"] - 5.0) < 1e-9


def test_prediction_with_no_nearby_actual_gets_none(tmp_path: Path):
    from tests.api_conftest import build_test_ctx

    ctx = build_test_ctx(tmp_path)
    now = time.time()
    # Far-future target -- nothing observed anywhere near it.
    _insert_forecast(ctx, target_timestamp=now + 10_000.0, predicted_value=42.0)

    from aiperfopt.api.app import create_app
    from fastapi.testclient import TestClient

    with TestClient(create_app(ctx)) as client:
        r = client.get("/reporting/predicted-vs-actual", params={"metric_name": "cpu_percent"})
        points = r.json()
        assert len(points) == 1
        assert points[0]["actual_value"] is None
        assert points[0]["error"] is None


def test_horizon_label_filter(tmp_path: Path):
    from tests.api_conftest import build_test_ctx

    ctx = build_test_ctx(tmp_path)
    now = time.time()
    _insert_forecast(ctx, target_timestamp=now + 40.0, predicted_value=10.0, horizon_label="short")
    _insert_forecast(ctx, target_timestamp=now + 200.0, predicted_value=20.0, horizon_label="medium")

    from aiperfopt.api.app import create_app
    from fastapi.testclient import TestClient

    with TestClient(create_app(ctx)) as client:
        r = client.get(
            "/reporting/predicted-vs-actual",
            params={"metric_name": "cpu_percent", "horizon_label": "medium"},
        )
        points = r.json()
        assert len(points) == 1
        assert points[0]["horizon_label"] == "medium"


def test_valid_metric_names_endpoint(tmp_path: Path):
    with build_test_client(tmp_path, with_history=False) as client:
        r = client.get("/reporting/metrics")
        assert r.status_code == 200
        names = r.json()
        assert "cpu_percent" in names
        assert names == sorted(names)


def test_summary_requires_period_or_explicit_window(tmp_path: Path):
    with build_test_client(tmp_path, with_history=False) as client:
        r = client.get("/reporting/summary")
        assert r.status_code == 400


def test_summary_with_explicit_window_returns_all_sections(tmp_path: Path):
    from tests.api_conftest import build_test_ctx

    ctx = build_test_ctx(tmp_path)
    samples = rising_cpu_samples_now(n=50, interval=2.0)
    from aiperfopt.storage import insert_system_metric_samples_batch

    insert_system_metric_samples_batch(ctx.storage_engine, samples)

    start = samples[0].timestamp
    end = samples[-1].timestamp

    from aiperfopt.api.app import create_app
    from fastapi.testclient import TestClient

    with TestClient(create_app(ctx)) as client:
        r = client.get("/reporting/summary", params={"start": start, "end": end})
        assert r.status_code == 200
        body = r.json()
        assert body["start"] == start
        assert body["end"] == end
        assert any(m["metric_name"] == "cpu_percent" for m in body["system_metrics"])
        assert body["prediction_accuracy"] == []
        assert body["contention"]["prediction_count"] == 0
        assert body["diagnosis"]["total_count"] == 0
        assert "cpu_bound_single_process" in body["diagnosis"]["counts_by_category"]
        assert body["recommendation"]["total_count"] == 0


def test_summary_with_period(tmp_path: Path):
    with build_test_client(tmp_path, with_history=False) as client:
        r = client.get("/reporting/summary", params={"period": "daily"})
        assert r.status_code == 200
        body = r.json()
        assert abs((body["end"] - body["start"]) - 24 * 3600.0) < 5.0


def test_summary_download_sets_content_disposition(tmp_path: Path):
    with build_test_client(tmp_path, with_history=False) as client:
        r = client.get("/reporting/summary", params={"period": "daily", "download": True})
        assert r.status_code == 200
        assert "attachment" in r.headers.get("content-disposition", "")
        assert r.json()["start"] is not None

