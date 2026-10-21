"""Generate the module table from the live test suite.

Counts are never hand-typed: each row's numbers come from asking pytest to
*collect* (not run) the test files that belong to that module, filtered by
the `behavioural`/`plumbing` markers every test carries (see
`pyproject.toml`'s `[tool.pytest.ini_options]`). Re-run this after any
test is added, removed, or re-marked.

    python scripts/generate_module_table.py

Writes `docs/module-table.md`.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Module:
    name: str
    responsibility: str
    source_path: str
    test_globs: tuple[str, ...]

    def test_paths(self) -> tuple[str, ...]:
        paths = sorted({str(p.relative_to(REPO_ROOT)) for g in self.test_globs for p in REPO_ROOT.glob(g)})
        if not paths:
            raise RuntimeError(f"{self.name}: no test files match {self.test_globs}")
        return tuple(paths)


MODULES: tuple[Module, ...] = (
    Module("Data Storage", "Schema, typed records, chronological reads, batched writes, roll-up and retention",
           "src/aiperfopt/storage/", ("tests/test_storage_*.py",)),
    Module("Monitoring + Ingestion", "Threaded system/process polling, bounded queue, batched ordered writes, overhead budget",
           "src/aiperfopt/monitoring/", ("tests/test_monitoring_*.py", "tests/test_overhead.py")),
    Module("Feature Engineering", "One function for training and live features: lags, rates, rolling windows, process activity",
           "src/aiperfopt/features/", ("tests/test_features_engine.py", "tests/test_features_process_features.py",
                                       "tests/test_features_storage_bridge.py")),
    Module("Preprocessing", "Validity and warm-up row filters; log-modulus, zero-variance and correlation column filters",
           "src/aiperfopt/features/preprocessing.py", ("tests/test_features_preprocessing.py",)),
    Module("Forecasting", "LightGBM forecaster, persistence / moving-average / Holt baselines, walk-forward evaluation",
           "src/aiperfopt/forecasting/", ("tests/test_forecasting_*.py",)),
    Module("Contention + Anomaly", "Logistic contention classifier with fitted threshold; Isolation Forest anomaly scoring",
           "src/aiperfopt/contention/, src/aiperfopt/anomaly/", ("tests/test_contention_*.py", "tests/test_anomaly_*.py")),
    Module("Diagnosis / XAI", "Cause taxonomy, SHAP attribution, technical and plain-language text",
           "src/aiperfopt/insight/", ("tests/test_insight_diagnosis.py", "tests/test_insight_explainer.py",
                                      "tests/test_insight_taxonomy.py", "tests/test_insight_text.py")),
    Module("Recommendation", "Cause -> curated action catalogue and lookup",
           "src/aiperfopt/insight/", ("tests/test_insight_catalog.py", "tests/test_insight_recommendations.py")),
    Module("Model Lifecycle", "Deployment gate, candidate-vs-active comparison, confirm / rollback, real-usage training data",
           "src/aiperfopt/lifecycle/", ("tests/test_lifecycle_*.py",)),
    Module("Model Comparison + Evaluation", "LightGBM vs Random Forest vs Ridge, ablation, evidence pack",
           "src/aiperfopt/evaluation/", ("tests/test_evaluation_*.py",)),
    Module("API Layer", "FastAPI app, live-inference loop, WebSocket, REST routers",
           "src/aiperfopt/api/", ("tests/test_api_*.py", "tests/test_live_inference_plumbing.py")),
    Module("Reporting / Analytics", "Predicted-vs-actual join, daily/weekly report assembly",
           "src/aiperfopt/reporting/", ("tests/test_reporting_*.py",)),
    Module("Web Dashboard", "Browser dashboard served by the API: Live, Accuracy, Reports, Compare, Model views",
           "src/aiperfopt/web/", ("tests/test_web_*.py",)),
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
    result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True)
    out = result.stdout
    if NO_TESTS_RE.search(out):
        return 0
    m = COLLECTED_RE.search(out)
    if not m:
        raise RuntimeError(
            f"Could not parse pytest collect-only output for {test_paths} (marker={marker}):\n"
            f"{out}\n{result.stderr}"
        )
    return int(m.group(1))


def main() -> None:
    rows = []
    total_behavioural = 0
    total_plumbing = 0
    for module in MODULES:
        total = _collect_count(module.test_paths(), None)
        behavioural = _collect_count(module.test_paths(), "behavioural")
        plumbing = _collect_count(module.test_paths(), "plumbing")
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
        "# Module Table",
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
        f"**Module total: {module_sum} tests** "
        f"({total_behavioural} behavioural, {total_plumbing} plumbing).",
        "",
        f"**Full suite: {suite_total} tests.** The difference "
        f"({suite_total - module_sum}) is cross-cutting infrastructure not "
        "owned by a single module -- CLI, layered config, structured "
        "logging, paths, readiness, replay and packaged-resources checks -- "
        "each exercised by its own test file under `tests/`. The dashboard's "
        "JavaScript helpers have a further suite run with "
        "`node --test src/aiperfopt/web/static/js/`.",
        "",
        "Any module whose Behavioural column is 0 has no test asserting a",
        "*correct answer* for that subsystem and should be treated as",
        "incomplete regardless of its total test count.",
        "",
    ]

    out_path = REPO_ROOT / "docs" / "module-table.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path} ({module_sum}/{suite_total} tests accounted for in the table)")


if __name__ == "__main__":
    main()
