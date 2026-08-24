from __future__ import annotations

from pathlib import Path

from aiperfopt.frontend.api_client import ApiClient
from aiperfopt.frontend.config import FrontendSettings
from aiperfopt.frontend.widgets.retraining_panel import RetrainingPanel
from aiperfopt.storage import insert_system_metric_samples_batch
from tests.api_conftest import build_test_ctx, rising_cpu_samples_now
from tests.frontend.server_fixture import running_backend


def _settings_for(base_url: str) -> FrontendSettings:
    host, port = base_url.replace("http://", "").split(":")
    return FrontendSettings(api_host=host, api_port=int(port))


def test_run_and_confirm_full_round_trip(qtbot, tmp_path: Path):
    ctx = build_test_ctx(tmp_path)
    samples = rising_cpu_samples_now(n=300, interval=2.0)
    insert_system_metric_samples_batch(ctx.storage_engine, samples)

    with running_backend(tmp_path, ctx=ctx) as (base_url, _ctx):
        client = ApiClient(settings=_settings_for(base_url))
        panel = RetrainingPanel(client)
        qtbot.addWidget(panel)

        panel.refresh_active_version()
        qtbot.waitUntil(lambda: panel._active_version_label.text() == "--", timeout=5000)

        panel._new_version_edit.setText("v1")
        panel._data_source_combo.setCurrentText("real_usage")
        panel._target_metrics_edit.setText("cpu_percent")
        panel._on_run_clicked()

        qtbot.waitUntil(lambda: panel._confirm_button.isEnabled(), timeout=30000)
        assert "v1" in panel._recommendation_label.text()

        panel._confirm(True)
        # Generous timeouts: under heavy machine load (e.g. the full test
        # suite running concurrently) two sequential REST round-trips here
        # (confirm, then the follow-up refresh_active_version() it
        # triggers) can occasionally take longer than a lightly-loaded
        # box would suggest -- this is test-environment contention, not
        # NFR-3.3's <=500ms production latency target, which is measured
        # against an idle backend.
        qtbot.waitUntil(lambda: "deployed" in panel._status_label.text(), timeout=20000)
        qtbot.waitUntil(lambda: panel._active_version_label.text() == "v1", timeout=20000)
        assert panel._confirm_button.isEnabled() is False

        client.close()


def test_run_without_a_version_name_shows_a_hint(qtbot, tmp_path: Path):
    with running_backend(tmp_path) as (base_url, _ctx):
        client = ApiClient(settings=_settings_for(base_url))
        panel = RetrainingPanel(client)
        qtbot.addWidget(panel)

        panel._new_version_edit.setText("")
        panel._on_run_clicked()
        assert "Enter a new model version name" in panel._status_label.text()

        client.close()
