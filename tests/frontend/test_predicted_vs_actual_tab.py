from __future__ import annotations

from pathlib import Path

import pytest

from aiperfopt.frontend.api_client import ApiClient
from aiperfopt.frontend.config import FrontendSettings
from aiperfopt.frontend.widgets.predicted_vs_actual_tab import PredictedVsActualTab
from aiperfopt.storage import insert_forecast_prediction, insert_system_metric_samples_batch
from aiperfopt.storage.records import ForecastPredictionRecord
from tests.api_conftest import build_test_ctx, rising_cpu_samples_now
from tests.frontend.server_fixture import running_backend


pytestmark = pytest.mark.plumbing


def _settings_for(base_url: str) -> FrontendSettings:
    host, port = base_url.replace("http://", "").split(":")
    return FrontendSettings(api_host=host, api_port=int(port))


def test_refresh_populates_chart(qtbot, tmp_path: Path):
    import time

    ctx = build_test_ctx(tmp_path)
    samples = rising_cpu_samples_now(n=50, interval=2.0)
    insert_system_metric_samples_batch(ctx.storage_engine, samples)

    target = samples[25].timestamp
    insert_forecast_prediction(
        ctx.storage_engine,
        ForecastPredictionRecord(
            generated_at=target - 30.0, target_timestamp=target, metric_name="cpu_percent",
            horizon_label="short", horizon_seconds=30,
            predicted_value=samples[25].cpu_percent + 3.0, model_version="v1", confidence=0.7,
        ),
    )

    with running_backend(tmp_path, ctx=ctx) as (base_url, _ctx):
        client = ApiClient(settings=_settings_for(base_url))
        tab = PredictedVsActualTab(client)
        qtbot.addWidget(tab)

        tab._lookback_spin.setValue(60 * 24)  # generous window covering the fixture's timestamps
        tab.refresh()

        qtbot.waitUntil(lambda: tab._chart._predicted_series.count() == 1, timeout=5000)
        assert "1 points" in tab._status_label.text()

        client.close()


def test_refresh_only_requests_the_selected_horizon(qtbot, tmp_path: Path):
    """Regression: leaving horizon unfiltered used to join short- and
    medium-horizon forecasts (made at the same tick but targeting very
    different future moments and typically very different accuracy) into
    one series, producing a jagged line with no coherent story. The tab
    must pass a specific horizon_label through to the API rather than
    leaving it unset."""
    ctx = build_test_ctx(tmp_path)
    samples = rising_cpu_samples_now(n=50, interval=2.0)
    insert_system_metric_samples_batch(ctx.storage_engine, samples)

    target = samples[25].timestamp
    insert_forecast_prediction(
        ctx.storage_engine,
        ForecastPredictionRecord(
            generated_at=target - 30.0, target_timestamp=target, metric_name="cpu_percent",
            horizon_label="short", horizon_seconds=30,
            predicted_value=samples[25].cpu_percent + 3.0, model_version="v1", confidence=0.7,
        ),
    )
    insert_forecast_prediction(
        ctx.storage_engine,
        ForecastPredictionRecord(
            generated_at=target - 30.0, target_timestamp=target + 60.0, metric_name="cpu_percent",
            horizon_label="medium", horizon_seconds=90,
            predicted_value=samples[25].cpu_percent - 10.0, model_version="v1", confidence=0.4,
        ),
    )

    with running_backend(tmp_path, ctx=ctx) as (base_url, _ctx):
        client = ApiClient(settings=_settings_for(base_url))
        tab = PredictedVsActualTab(client)
        qtbot.addWidget(tab)

        tab._lookback_spin.setValue(60 * 24)
        tab._horizon_combo.setCurrentText("short")
        tab.refresh()

        qtbot.waitUntil(lambda: "points" in tab._status_label.text(), timeout=5000)
        assert tab._chart._predicted_series.count() == 1
        assert "short-horizon" in tab._chart._chart.title()

        client.close()


def test_refresh_with_no_data_shows_empty_message(qtbot, tmp_path: Path):
    with running_backend(tmp_path) as (base_url, _ctx):
        client = ApiClient(settings=_settings_for(base_url))
        tab = PredictedVsActualTab(client)
        qtbot.addWidget(tab)

        tab.refresh()
        qtbot.waitUntil(lambda: tab._status_label.text() != "Loading...", timeout=5000)
        assert "No predictions" in tab._status_label.text()

        client.close()
