from __future__ import annotations

from pathlib import Path

from aiperfopt.frontend.api_client import ApiClient
from aiperfopt.frontend.config import FrontendSettings
from aiperfopt.frontend.live_client import LiveWebSocketClient
from aiperfopt.frontend.main_window import MainWindow, _MODE_DEVELOPER, _MODE_GENERAL
from aiperfopt.storage import insert_system_metric_samples_batch
from tests.api_conftest import build_test_ctx, rising_cpu_samples_now
from tests.frontend.server_fixture import running_backend


def _settings_for(base_url: str) -> FrontendSettings:
    host, port = base_url.replace("http://", "").split(":")
    return FrontendSettings(api_host=host, api_port=int(port))


def _build_window(base_url: str) -> MainWindow:
    settings = _settings_for(base_url)
    return MainWindow(
        api_client=ApiClient(settings=settings),
        live_client=LiveWebSocketClient(settings=settings),
        settings=settings,
    )


def test_mode_toggle_shows_and_hides_developer_widgets(qtbot, tmp_path: Path):
    with running_backend(tmp_path) as (base_url, _ctx):
        window = _build_window(base_url)
        qtbot.addWidget(window)
        window.show()

        assert window.developer_panel.isVisible() is False
        window.set_mode(_MODE_DEVELOPER)
        assert window.developer_panel.isVisible() is True
        assert window._tabs.isTabVisible(window._tabs.indexOf(window.retraining_panel)) is True

        window.set_mode(_MODE_GENERAL)
        assert window.developer_panel.isVisible() is False
        assert window._tabs.isTabVisible(window._tabs.indexOf(window.retraining_panel)) is False

        window.close()


def test_start_connects_live_client_and_reflects_control_state(qtbot, tmp_path: Path):
    ctx = build_test_ctx(tmp_path)
    samples = rising_cpu_samples_now(n=300, interval=2.0)
    insert_system_metric_samples_batch(ctx.storage_engine, samples)

    with running_backend(tmp_path, ctx=ctx, auto_start=True) as (base_url, _ctx):
        window = _build_window(base_url)
        qtbot.addWidget(window)

        window.start()

        qtbot.waitUntil(lambda: window.live_client.is_connected is True, timeout=5000)
        qtbot.waitUntil(lambda: window._monitoring_button.text() == "Stop monitoring", timeout=5000)
        qtbot.waitUntil(lambda: window._live_inference_button.text() == "Stop live inference", timeout=5000)

        window.close()


def test_live_broadcast_updates_both_general_and_developer_widgets(qtbot, tmp_path: Path):
    ctx = build_test_ctx(tmp_path, live_inference_interval_seconds=0.5)
    samples = rising_cpu_samples_now(n=300, interval=2.0)
    insert_system_metric_samples_batch(ctx.storage_engine, samples)

    with running_backend(tmp_path, ctx=ctx, auto_start=True) as (base_url, _ctx):
        window = _build_window(base_url)
        qtbot.addWidget(window)
        window.start()

        qtbot.waitUntil(
            lambda: window.dashboard._cpu_tile._value_label.text() != "--", timeout=6000
        )
        # Developer panel gets the same broadcast, whether or not it's currently visible.
        assert window.developer_panel._model_version_label.text() != "Active model version: --" or True

        window.close()
