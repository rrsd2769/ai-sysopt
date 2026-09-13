from __future__ import annotations

from pathlib import Path

import pytest

from aiperfopt.contention.training_pipeline import ContentionTrainingSummary
from aiperfopt.forecasting.training_pipeline import MetricHorizonOutcome, TrainingRunSummary
from aiperfopt.evaluation.ablation import AblationReport, ArmResult
from aiperfopt.evaluation.report import write_ablation_report


def _fake_outcome(metric: str, horizon: str, feature_importance=None) -> MetricHorizonOutcome:
    return MetricHorizonOutcome(
        metric_name=metric, horizon_label=horizon, horizon_seconds=30,
        walk_forward_summary={
            "lightgbm": {"mae": 1.0, "rmse": 1.5, "directional_accuracy": 0.7, "n_folds": 2},
            "naive_persistence": {"mae": 2.0, "rmse": 2.5, "directional_accuracy": 0.5, "n_folds": 2},
        },
        selected_forecaster="lightgbm", holdout_mae=1.1, holdout_rmse=1.6, holdout_directional_accuracy=0.65,
        artifact_path="models/cpu_percent/short/test", feature_importance=feature_importance,
    )


def _fake_report() -> AblationReport:
    v1_contention = ContentionTrainingSummary(
        model_version="test-v1", n_training_rows=100,
        holdout_metrics={"precision": 0.5, "recall": 0.4, "f1": 0.44, "pr_auc": 0.45},
        base_rate=0.1, decision_threshold=0.6,
        threshold_rule_baseline={"precision": 0.3, "recall": 0.3, "f1": 0.3},
        holdout_probabilities=[0.1, 0.9, 0.2, 0.8], holdout_labels=[0, 1, 0, 1],
    )
    v2_contention = ContentionTrainingSummary(
        model_version="test-v2", n_training_rows=100,
        holdout_metrics={"precision": 0.7, "recall": 0.6, "f1": 0.65, "pr_auc": 0.68},
        base_rate=0.1, decision_threshold=0.55,
        threshold_rule_baseline={"precision": 0.3, "recall": 0.3, "f1": 0.3},
        holdout_probabilities=[0.05, 0.95, 0.3, 0.85], holdout_labels=[0, 1, 0, 1],
    )

    v1 = ArmResult(
        arm="v1", model_version="test-v1",
        forecast_summary=TrainingRunSummary(model_version="test-v1", outcomes=[_fake_outcome("cpu_percent", "short")]),
        contention_summary=v1_contention,
    )
    v2 = ArmResult(
        arm="v2", model_version="test-v2",
        forecast_summary=TrainingRunSummary(
            model_version="test-v2",
            outcomes=[_fake_outcome(
                "cpu_percent", "short",
                feature_importance={"cpu_percent_lag_10s": 0.2, "proc_young_cpu_sum_lag_10s": 0.5},
            )],
        ),
        contention_summary=v2_contention,
    )

    return AblationReport(
        dataset_summary={
            "training_rows": 100, "holdout_rows": 25, "total_rows": 125,
            "hours_spanned": 1.5, "contention_base_rate": 0.1,
        },
        arms={"v1": v1, "v2": v2},
    )


pytestmark = pytest.mark.plumbing


def test_write_ablation_report_produces_every_artifact(tmp_path: Path):
    report = _fake_report()
    out_dir = tmp_path / "results"

    written = write_ablation_report(report, out_dir, overhead_warmup_seconds=0.1, overhead_run_seconds=0.3)

    expected = {
        "dataset_summary.txt", "ablation.csv", "baseline_comparison.csv",
        "pr_curve_contention.png", "confusion_matrix.png", "feature_importance.png", "overhead.txt",
    }
    written_names = {p.name for p in written}
    assert expected <= written_names

    for name in expected:
        path = out_dir / name
        assert path.exists()
        assert path.stat().st_size > 0


def test_ablation_csv_has_one_row_per_arm_target(tmp_path: Path):
    import csv

    report = _fake_report()
    out_dir = tmp_path / "results"
    write_ablation_report(report, out_dir, overhead_warmup_seconds=0.1, overhead_run_seconds=0.3)

    with (out_dir / "ablation.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    targets = {(row["arm"], row["target"]) for row in rows}
    assert ("v1", "contention") in targets
    assert ("v2", "contention") in targets
    assert ("v1", "forecast_cpu_percent_short") in targets
    assert ("v2", "forecast_cpu_percent_short") in targets


def test_dataset_summary_txt_contains_key_figures(tmp_path: Path):
    report = _fake_report()
    out_dir = tmp_path / "results"
    write_ablation_report(report, out_dir, overhead_warmup_seconds=0.1, overhead_run_seconds=0.3)

    text = (out_dir / "dataset_summary.txt").read_text(encoding="utf-8")
    assert "training_rows" in text
    assert "125" in text or "100" in text
