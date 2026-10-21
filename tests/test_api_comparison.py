from __future__ import annotations

from pathlib import Path

import pytest

from tests.api_conftest import build_test_client

HOLDOUT = """target,model,mae,rmse,directional_accuracy,fit_seconds,latency_ms,size_kb
cpu_percent_short,LightGBM,2.95,4.88,0.54,6.0,1.75,139.0
cpu_percent_short,Persistence (reference),3.02,5.48,0.10,,,
"""
CV = """target,fold,model,mae,rmse,directional_accuracy
cpu_percent_short,0,LightGBM,3.82,5.82,0.53
"""


@pytest.mark.plumbing
def test_404_until_a_comparison_has_been_run(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        client.app.state.results_dir = tmp_path / "results"
        r = client.get("/comparison/models")
        assert r.status_code == 404
        assert "aiperfopt compare" in r.json()["detail"]


@pytest.mark.behavioural
def test_returns_saved_results_with_numbers_parsed(tmp_path: Path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "model_comparison.csv").write_text(HOLDOUT)
    (results / "model_comparison_cv.csv").write_text(CV)
    with build_test_client(tmp_path) as client:
        client.app.state.results_dir = results
        body = client.get("/comparison/models").json()
        forecasting = client.app.state.ctx.settings.forecasting

    lgbm, persistence = body["holdout"]
    assert lgbm["model"] == "LightGBM" and lgbm["mae"] == 2.95 and lgbm["latency_ms"] == 1.75
    assert persistence["fit_seconds"] is None
    assert body["cv"] == [{"target": "cpu_percent_short", "fold": 0, "model": "LightGBM",
                           "mae": 3.82, "rmse": 5.82, "directional_accuracy": 0.53}]
    assert body["regimes"] == []
    assert body["horizon_seconds"] == {
        "short": forecasting.short_horizon_target_seconds,
        "medium": forecasting.medium_horizon_target_seconds,
    }
