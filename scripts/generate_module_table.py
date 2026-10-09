"""Generate the Phase 10 module table from the live test suite.

Counts are never hand-typed: each row's numbers come from asking pytest to
*collect* (not run) the test files that belong to that module, filtered by
the `behavioural`/`plumbing` markers every test in the suite now carries
(see `pyproject.toml`'s `[tool.pytest.ini_options]` and
`docs/plans/2026-09-12-presentable-predictive-monitor.md`'s Testing
Strategy section). Re-run this after any test is added, removed, or
re-marked; it will reflect the change automatically.

    python scripts/generate_module_table.py

Writes `docs/phase10-module-table.md`.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The eleven subsystems named by the project's roadmap (README.md's
# "Roadmap" section, items 1-11) -- item 0 (Project Foundation) and
# cross-cutting infrastructure (config, applog, common, replay, evaluation
# harness, CLI) are deliberately excluded, matching the plan's own
# statement that replay and process-feature extraction added no new
# module ("Same eleven subsystems").
@dataclass(frozen=True)
class Module:
    name: str
    responsibility: str
    source_path: str
    test_globs: tuple[str, ...]


MODULES: tuple[Module, ...] = (
    Module(
        "Data Storage",
        "Schema, typed records, chronological reads, batched writes, downsampling/retention",
        "src/aiperfopt/storage/",
        ("tests/test_storage_engine.py", "tests/test_storage_reader_writer.py",
         "tests/test_storage_rollup.py", "tests/test_storage_schema.py"),
    ),
    Module(
        "Monitoring Engine + Data Ingestion",
        "Threaded system/process polling, launch/terminate detection, batched ordered storage writes",
        "src/aiperfopt/monitoring/, src/aiperfopt/ingestion/",
        ("tests/test_monitoring_collectors.py", "tests/test_monitoring_engine.py",
         "tests/test_monitoring_process_tracker.py", "tests/test_ingestion_consumer.py"),
    ),
    Module(
        "Feature Engineering",
        "Single source of truth for lag/rolling-window features, system + process, live and backfill",
        "src/aiperfopt/features/",
        ("tests/test_features_engine.py", "tests/test_features_process_features.py",
         "tests/test_features_storage_bridge.py"),
    ),
    Module(
        "Forecasting Models + Evaluation Harness",
        "LightGBM-backed metric forecaster, baselines, walk-forward validation, persistence/deployment",
        "src/aiperfopt/forecasting/",
        ("tests/test_forecasting_baselines.py", "tests/test_forecasting_dataset.py",
         "tests/test_forecasting_evaluation.py", "tests/test_forecasting_model.py",
         "tests/test_forecasting_persistence.py", "tests/test_forecasting_training_pipeline.py"),
    ),
    Module(
        "Contention Prediction + Anomaly Detection",
        "Contention classifier + contributing processes; Isolation Forest anomaly scoring",
        "src/aiperfopt/contention/, src/aiperfopt/anomaly/",
        ("tests/test_contention_dataset.py", "tests/test_contention_evaluation.py",
         "tests/test_contention_model.py", "tests/test_contention_persistence.py",
         "tests/test_contention_prediction.py", "tests/test_contention_processes.py",
         "tests/test_contention_training_pipeline.py", "tests/test_anomaly_calibration.py",
         "tests/test_anomaly_detector.py", "tests/test_anomaly_persistence_and_notifications.py"),
    ),
    Module(
        "Diagnosis / Explainable AI",
        "Category taxonomy, tiered lightweight/SHAP attribution, technical + plain-language text",
        "src/aiperfopt/diagnosis/",
        ("tests/test_diagnosis_explainer.py", "tests/test_diagnosis_service.py",
         "tests/test_diagnosis_taxonomy.py", "tests/test_diagnosis_text.py"),
    ),
    Module(
        "Recommendation Engine",
        "Category -> curated actions catalog and lookup",
        "src/aiperfopt/recommendation/",
        ("tests/test_recommendation_catalog.py", "tests/test_recommendation_service.py"),
    ),
    Module(
        "Model Retraining",
        "Real-usage retraining, candidate-vs-active comparison, confirm/rollback deployment",
        "src/aiperfopt/retraining/",
        ("tests/test_retraining_active_version.py", "tests/test_retraining_comparison.py",
         "tests/test_retraining_deployment.py", "tests/test_retraining_pipeline.py",
         "tests/test_retraining_real_usage_data.py"),
    ),
    Module(
        "API Layer",
        "FastAPI app, live-inference loop, WebSocket fan-out, REST history/retraining/reporting routers",
        "src/aiperfopt/api/",
        ("tests/test_api_history.py", "tests/test_api_live.py", "tests/test_api_reporting.py",
         "tests/test_api_retraining.py", "tests/test_api_system.py",
         "tests/test_live_inference_plumbing.py"),
    ),
    Module(
        "Frontend",
        "PySide6 desktop UI: dashboard, live chart/contention strip, Predicted vs. Actual, Reporting, Retraining panels",
        "src/aiperfopt/frontend/",
        ("tests/frontend/",),
    ),
    Module(
        "Reporting / Analytics",
        "Predicted-vs-actual join, period/window report assembly (accuracy, contention, diagnosis counts)",
        "src/aiperfopt/reporting/",
        ("tests/test_reporting_predicted_vs_actual.py", "tests/test_reporting_summary.py"),
    ),
)

# Pytest prints "N tests collected" with no filter, or "N/M tests
# collected (K deselected)" when a marker filter (-m) excluded some --
# the leading N (not M) is the count we want in both cases.
COLLECTED_RE = re.compile(r"^(\d+)(?:/\d+)?\s+tests? collected", re.MULTILINE)
NO_TESTS_RE = re.compile(r"^no tests (ran|collected)", re.MULTILINE)


def _collect_count(test_paths: tuple[str, ...], marker: str | None) -> int:
    args = [
        sys.executable, "-m", "pytest", "--collect-only", "-q",
        "-p", "no:cacheprovider",
        *test_paths,
    ]
    if marker is not None:
        args += ["-m", marker]
    env_note = "QT_QPA_PLATFORM=offscreen"
    import os
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        args, cwd=REPO_ROOT, capture_output=True, text=True, env=env,
    )
    out = result.stdout
    if NO_TESTS_RE.search(out):
        return 0
    m = COLLECTED_RE.search(out)
    if not m:
        raise RuntimeError(
            f"Could not parse pytest collect-only output for {test_paths} (marker={marker}):\n"
            f"{out}\n{result.stderr}\n({env_note} was set)"
        )
    return int(m.group(1))


def main() -> None:
    rows = []
    total_behavioural = 0
    total_plumbing = 0
    for module in MODULES:
        total = _collect_count(module.test_globs, None)
        behavioural = _collect_count(module.test_globs, "behavioural")
        plumbing = _collect_count(module.test_globs, "plumbing")
        if behavioural + plumbing != total:
            raise RuntimeError(
                f"{module.name}: behavioural({behavioural}) + plumbing({plumbing}) "
                f"!= total({total}) -- a test is unmarked or double-marked."
            )
        total_behavioural += behavioural
        total_plumbing += plumbing
        rows.append((module, total, behavioural, plumbing))

    suite_total = _collect_count((), None)

    lines = [
        "# Module Table (Phase 10.2)",
        "",
        "Generated by `scripts/generate_module_table.py` -- do not hand-edit the",
        "counts below; re-run the script after any test changes. Counts come from",
        "`pytest --collect-only`, filtered by each test's `behavioural` or",
        "`plumbing` marker (`pyproject.toml`, `--strict-markers` enforced).",
        "",
        "**Behavioural**: asserts a computed value/decision/output is *correct*,",
        "not merely that code ran. **Plumbing**: asserts wiring, construction, or",
        "execution only.",
        "",
        "| # | Module | Responsibility | Source | Tests | Behavioural | Plumbing |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for i, (module, total, behavioural, plumbing) in enumerate(rows, start=1):
        lines.append(
            f"| {i} | {module.name} | {module.responsibility} | `{module.source_path}` "
            f"| {total} | {behavioural} | {plumbing} |"
        )

    module_sum = sum(r[1] for r in rows)
    lines += [
        "",
        f"**Eleven-module total: {module_sum} tests** "
        f"({total_behavioural} behavioural, {total_plumbing} plumbing).",
        "",
        f"**Full suite: {suite_total} tests.** The difference "
        f"({suite_total - module_sum}) is cross-cutting infrastructure not "
        "owned by a single module -- CLI, layered config, structured "
        "logging, readiness, replay, the "
        "evaluation/ablation harness, overhead budget, end-to-end pipeline "
        "integration, and packaged-resources checks -- each already "
        "exercised directly by its own test file under `tests/`.",
        "",
        "Any module whose Behavioural column is 0 has no test asserting a",
        "*correct answer* for that subsystem and should be treated as",
        "incomplete regardless of its total test count.",
        "",
    ]

    out_path = REPO_ROOT / "docs" / "phase10-module-table.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path} ({module_sum}/{suite_total} tests accounted for in the table)")


if __name__ == "__main__":
    main()
