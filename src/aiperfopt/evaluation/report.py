"""
Writes Phase 5's Review 2 evidence pack from an
:class:`aiperfopt.evaluation.ablation.AblationReport` — the artifacts named
in ``docs/plans/2026-09-12-presentable-predictive-monitor.md``'s Phase 5
section: ``dataset_summary.txt``, ``ablation.csv``,
``baseline_comparison.csv``, ``pr_curve_contention.png``,
``confusion_matrix.png``, ``feature_importance.png``, ``overhead.txt``.

Matplotlib-backed plots (via scikit-learn's ``*Display`` helpers, which
require it) — the ``Agg`` backend is forced so this works headlessly on a
machine with no display, which a CLI command must assume.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 (must follow matplotlib.use)
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay  # noqa: E402

from aiperfopt.monitoring.overhead import measure_pipeline_overhead  # noqa: E402
from aiperfopt.evaluation.ablation import AblationReport  # noqa: E402

_TOP_N_FEATURES = 20


def _write_dataset_summary(report: AblationReport, out_dir: Path) -> None:
    lines = ["Phase 5 ablation — dataset summary", "=" * 40]
    for key, value in report.dataset_summary.items():
        lines.append(f"{key:<24}: {value}")

    v2 = report.arms.get("v2")
    contention = v2.contention_summary if v2 is not None else None
    if contention is None or not contention.holdout_probabilities:
        lines.append("")
        lines.append(
            "note: no contention classifier was trained on this window (labels came back "
            "single-class -- reported honestly rather than fabricated); "
            "pr_curve_contention.png and confusion_matrix.png are not produced."
        )

    (out_dir / "dataset_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ablation_csv(report: AblationReport, out_dir: Path) -> None:
    fieldnames = ["arm", "target", "pr_auc", "f1", "precision", "recall", "mae", "mae_persistence"]
    rows: list[dict] = []

    for arm_name, arm in report.arms.items():
        contention = arm.contention_summary
        if contention is not None and contention.holdout_metrics:
            hm = contention.holdout_metrics
            rows.append({
                "arm": arm_name, "target": "contention",
                "pr_auc": hm.get("pr_auc"), "f1": hm.get("f1"),
                "precision": hm.get("precision"), "recall": hm.get("recall"),
                "mae": None, "mae_persistence": None,
            })

        for outcome in arm.forecast_summary.outcomes:
            persistence_mae = outcome.walk_forward_summary.get("naive_persistence", {}).get("mae")
            rows.append({
                "arm": arm_name, "target": f"forecast_{outcome.metric_name}_{outcome.horizon_label}",
                "pr_auc": None, "f1": None, "precision": None, "recall": None,
                "mae": outcome.holdout_mae, "mae_persistence": persistence_mae,
            })

    with (out_dir / "ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_baseline_comparison_csv(report: AblationReport, out_dir: Path) -> None:
    """v2 (system + process) arm's model vs every baseline, per target."""
    fieldnames = ["target", "model", "mae", "rmse", "directional_accuracy", "precision", "recall", "f1", "pr_auc"]
    rows: list[dict] = []

    v2 = report.arms.get("v2")
    if v2 is not None:
        for outcome in v2.forecast_summary.outcomes:
            target = f"{outcome.metric_name}_{outcome.horizon_label}"
            for model_name, metrics in outcome.walk_forward_summary.items():
                rows.append({
                    "target": target, "model": model_name,
                    "mae": metrics.get("mae"), "rmse": metrics.get("rmse"),
                    "directional_accuracy": metrics.get("directional_accuracy"),
                    "precision": None, "recall": None, "f1": None, "pr_auc": None,
                })

        contention = v2.contention_summary
        if contention is not None:
            if contention.holdout_metrics:
                hm = contention.holdout_metrics
                rows.append({
                    "target": "contention", "model": "contention_classifier",
                    "mae": None, "rmse": None, "directional_accuracy": None,
                    "precision": hm.get("precision"), "recall": hm.get("recall"),
                    "f1": hm.get("f1"), "pr_auc": hm.get("pr_auc"),
                })
            if contention.threshold_rule_baseline:
                tr = contention.threshold_rule_baseline
                rows.append({
                    "target": "contention", "model": "threshold_rule",
                    "mae": None, "rmse": None, "directional_accuracy": None,
                    "precision": tr.get("precision"), "recall": tr.get("recall"),
                    "f1": tr.get("f1"), "pr_auc": None,
                })

    with (out_dir / "baseline_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_pr_curve_and_confusion_matrix(report: AblationReport, out_dir: Path) -> None:
    v2 = report.arms.get("v2")
    contention = v2.contention_summary if v2 is not None else None
    if contention is None or not contention.holdout_probabilities or not contention.holdout_labels:
        return

    labels = contention.holdout_labels
    probabilities = contention.holdout_probabilities
    threshold = contention.decision_threshold if contention.decision_threshold is not None else 0.5

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(labels, probabilities, ax=ax)
    hm = contention.holdout_metrics or {}
    if "recall" in hm and "precision" in hm:
        ax.scatter([hm["recall"]], [hm["precision"]], color="red", zorder=5,
                   label=f"fitted threshold={threshold:.2f}")
        ax.legend()
    ax.set_title("Contention Prediction — Precision-Recall (v2, system+process)")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve_contention.png")
    plt.close(fig)

    predictions = [1 if p >= threshold else 0 for p in probabilities]
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(labels, predictions, ax=ax)
    ax.set_title(f"Contention Prediction — Confusion Matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png")
    plt.close(fig)


def _write_feature_importance(report: AblationReport, out_dir: Path) -> None:
    v2 = report.arms.get("v2")
    if v2 is None:
        return

    combined: dict[str, float] = {}
    for outcome in v2.forecast_summary.outcomes:
        if not outcome.feature_importance:
            continue
        for feature, value in outcome.feature_importance.items():
            combined[feature] = combined.get(feature, 0.0) + value

    if not combined:
        return

    top = sorted(combined.items(), key=lambda item: item[1], reverse=True)[:_TOP_N_FEATURES]
    top.reverse()  # horizontal bar chart reads top-to-bottom as most-to-least important
    names = [name for name, _ in top]
    values = [value for _, value in top]
    colors = ["#d95f02" if name.startswith("proc_") else "#1b9e77" for name in names]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    ax.barh(names, values, color=colors)
    ax.set_xlabel("Mean |SHAP value| (summed across metric/horizon)")
    ax.set_title("Feature importance — v2 (system+process) forecasters")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png")
    plt.close(fig)


def _write_overhead(out_dir: Path, warmup_seconds: float, run_seconds: float) -> None:
    with tempfile.TemporaryDirectory(prefix="aiperfopt-overhead-") as tmp:
        measurement = measure_pipeline_overhead(Path(tmp), warmup_seconds=warmup_seconds, run_seconds=run_seconds)
    lines = [
        "Phase 2/5 — monitor overhead (isolated measurement, not the live collector)",
        "=" * 60,
        f"self CPU            : {measurement.cpu_percent:.2f}% (budget < {measurement.cpu_budget_percent}%)",
        f"database growth     : {measurement.db_growth_bytes_per_hour / (1024 * 1024):.2f} MB/hour "
        f"(ceiling < {measurement.db_growth_ceiling_bytes_per_hour / (1024 * 1024):.0f} MB/hour)",
    ]
    (out_dir / "overhead.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation_report(
    report: AblationReport,
    out_dir: Path,
    overhead_warmup_seconds: float = 2.0,
    overhead_run_seconds: float = 8.0,
) -> list[Path]:
    """Write every Phase 5 evidence-pack artifact into ``out_dir``.

    ``overhead_warmup_seconds``/``overhead_run_seconds`` control the
    isolated overhead measurement's duration (see
    :func:`aiperfopt.monitoring.overhead.measure_pipeline_overhead`) --
    shortened by tests so the suite isn't paying the real ~10s cost
    per test; a real report should use the defaults.

    Returns the list of files written, for a caller (the CLI) to print.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_dataset_summary(report, out_dir)
    _write_ablation_csv(report, out_dir)
    _write_baseline_comparison_csv(report, out_dir)
    _write_pr_curve_and_confusion_matrix(report, out_dir)
    _write_feature_importance(report, out_dir)
    _write_overhead(out_dir, overhead_warmup_seconds, overhead_run_seconds)

    return sorted(out_dir.iterdir())
