from __future__ import annotations

from pathlib import Path

import pytest

from aiperfopt.frontend.api_client import ApiClient
from aiperfopt.frontend.config import FrontendSettings
from aiperfopt.frontend.widgets.reporting_tab import ReportingTab
from aiperfopt.storage import insert_diagnosis, insert_system_metric_samples_batch
from aiperfopt.storage.records import DiagnosisRecord
from tests.api_conftest import build_test_ctx, rising_cpu_samples_now
from tests.frontend.server_fixture import running_backend


pytestmark = pytest.mark.plumbing


def _settings_for(base_url: str) -> FrontendSettings:
    host, port = base_url.replace("http://", "").split(":")
    return FrontendSettings(api_host=host, api_port=int(port))


def test_refresh_populates_all_sections(qtbot, tmp_path: Path):
    ctx = build_test_ctx(tmp_path)
    samples = rising_cpu_samples_now(n=30, interval=2.0)
    insert_system_metric_samples_batch(ctx.storage_engine, samples)
    insert_diagnosis(
        ctx.storage_engine,
        DiagnosisRecord(
            generated_at=samples[0].timestamp, category="cpu_bound_single_process",
            is_actionable=True, confidence=0.85, explanation_tier="shap",
            trigger_reason="high_confidence_contention", technical_explanation="x",
            plain_language_explanation="y", model_version="v1",
        ),
    )

    with running_backend(tmp_path, ctx=ctx) as (base_url, _ctx):
        client = ApiClient(settings=_settings_for(base_url))
        tab = ReportingTab(client)
        qtbot.addWidget(tab)

        tab._period_combo.setCurrentText("weekly")
        tab.refresh()

        qtbot.waitUntil(lambda: tab._system_metrics_tree.topLevelItemCount() > 0, timeout=5000)
        assert tab._diagnosis_tree.topLevelItemCount() == 1
        assert "1 diagnoses" in tab._diagnosis_label.text()
        assert "Report window" in tab._status_label.text()

        client.close()


def test_system_metrics_table_shows_only_percent_metrics_with_units(qtbot, tmp_path: Path):
    """Regression: the table used to include raw cumulative byte counters
    (disk/net) with no unit and no distinguishing formatting, and Min/Max
    were rendered as bare numbers with no "%" even for percent metrics
    while Avg had one -- confusing and inconsistent. Only the three
    self-explanatory percent metrics should render, all three columns
    formatted the same way."""
    ctx = build_test_ctx(tmp_path)
    samples = rising_cpu_samples_now(n=30, interval=2.0)
    insert_system_metric_samples_batch(ctx.storage_engine, samples)

    with running_backend(tmp_path, ctx=ctx) as (base_url, _ctx):
        client = ApiClient(settings=_settings_for(base_url))
        tab = ReportingTab(client)
        qtbot.addWidget(tab)

        tab.refresh()
        qtbot.waitUntil(lambda: tab._system_metrics_tree.topLevelItemCount() > 0, timeout=5000)

        tree = tab._system_metrics_tree
        shown_metrics = {tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())}
        assert shown_metrics == {"cpu_percent", "mem_percent", "swap_percent"}
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            for column in (1, 2, 3):  # Avg, Min, Max
                assert item.text(column) == "--" or item.text(column).endswith("%")
        assert "readings" in tab._status_label.text()

        client.close()


def test_refresh_with_no_data_shows_placeholder_prediction_row(qtbot, tmp_path: Path):
    with running_backend(tmp_path) as (base_url, _ctx):
        client = ApiClient(settings=_settings_for(base_url))
        tab = ReportingTab(client)
        qtbot.addWidget(tab)

        tab.refresh()
        qtbot.waitUntil(lambda: "Report window" in tab._status_label.text(), timeout=5000)
        assert tab._prediction_accuracy_tree.topLevelItemCount() == 1
        assert tab._diagnosis_tree.topLevelItemCount() == 0

        client.close()
