# Lean Codebase + Web Dashboard Implementation Plan

## Overview

Two goals, in this order:

1. **Make the codebase smaller and easier to follow.** One training source
   (real usage data) instead of two, model-lifecycle code in one package
   instead of four, docstrings that explain the code instead of citing a
   spec that isn't in the repo.
2. **Replace the PySide6 desktop UI with a well-designed web dashboard**,
   served by the FastAPI server that already exists. The current UI is
   light-theme hardcoded colors on a dark macOS window, with squashed
   charts and raw metric names.

The app must stay runnable at every phase boundary. The Qt UI is deleted
only after the web UI reaches feature parity (Phase 7).

## Current State Analysis

- `src/aiperfopt/`: ~90 modules, ~14k lines, 20 packages. **Every module
  is reachable from `aiperfopt.cli`** (verified by an AST import-graph
  walk). The only unreachable one is the untracked
  `features/preprocessing.py`, used only by `scripts/compare_models.py`.
  So this is not a dead-file problem. It is a duplication and sprawl
  problem:
  - **Two training data sources.** `synthetic.py` (225 lines) plus
    scenario generators in `forecasting/training_pipeline.py:113-140`,
    `run_retraining_on_synthetic_data` (`retraining/pipeline.py:176`),
    `aiperfopt train --source synthetic` (the *default*, `cli.py:558`),
    `RetrainRequest.data_source` plus synthetic-only fields
    (`api/schemas.py:186-196`), and a "synthetic" option in the Qt
    Retraining panel (`frontend/widgets/retraining_panel.py:61`). About
    73 h of real data now exists, and the paper never mentions synthetic
    data.
  - **Model lifecycle is scattered.** The quality gate and candidate
    comparison (`retraining/comparison.py`, 484 lines), deploy/rollback
    (`retraining/deployment.py`), the active-version pointer
    (`retraining/active_version.py`), loading deployed forecasters
    (`forecasting/deployment.py`) and the real-data split
    (`retraining/real_usage_data.py`) all answer one question: "which
    model is live and should a new one replace it". They sit in three
    packages.
  - **Insight is split for no reason.** `diagnosis/` and
    `recommendation/` always run back to back
    (`api/live_state.py` → diagnose → recommend).
  - **Producer is split.** `monitoring/`, `ingestion/consumer.py` (one
    class) and top-level `pipeline.py` together form one producer.
  - **Offline experiments live outside the package.**
    `scripts/compare_models.py` (the three-model comparison the deck and
    paper rely on) is not reachable from the CLI.
- **Docstring noise.** Pervasive references like `FR-11.1`,
  `Section 10.7`, `ADL-23`, `NFR-4.1`, `Phase 5`. These point at
  `PROJECT_SPEC.md` and plans no reader of the code has.
- **Unused symbols.** `common/exceptions.py`: `MetricSourceUnavailableError`,
  `IngestionError`, `FeatureEngineeringError`, `DiagnosisError`,
  `EvaluationError`, `APIError`, `FrontendError`, `ReportingError`.
  `api/dependencies.py`: `get_settings`, `get_recommendation_catalog`.
  `api/schemas.py`: `RecommendationActionOut`. These are confirmed
  unreferenced anywhere in `src/` or `tests/`.
- **Stale docs.** `PROJECT_HANDOFF.md` (1,581 lines, zip-upload era) and
  two completed plans in `docs/plans/`.
- **UI.** `src/aiperfopt/frontend/` (PySide6, ~1,900 lines) plus
  `tests/frontend/` (13 files). Every color is an inline `setStyleSheet`
  hex value. QtCharts are rendered on a white background. The CI
  frontend job only passes via `continue-on-error`.
- **API surface** (unchanged by this plan; the web UI uses it as-is):
  - `GET /health`, `/config`, `/readiness`
  - `GET|POST /monitoring/{status,start,stop}`, `/live/{status,start,stop}`
  - `GET /live/state`, `WS /ws/live` (`LiveStateOut`, `api/schemas.py`)
  - `GET /metrics/system`, `/forecasts`, `/contention`, `/diagnoses`,
    `/recommendations`, `/evaluations`, `/deployments[/latest]`, all
    taking `start`/`end`/`limit`
  - `GET /reporting/{predicted-vs-actual,summary,metrics}`
  - `GET /retraining/active-version`, `POST /retraining/{run,confirm,rollback}`
- **Baseline:** `pytest --ignore=tests/frontend -q` gives **501 passed in
  4m18s** (2026-10-09, including the untracked preprocessing tests).

## Desired End State

```
src/aiperfopt/
  cli.py  bootstrap.py  paths.py  readiness.py
  config/  applog/  common/
  monitoring/      collectors, engine, process_tracker, overhead, queue_items,
                   ingestion.py (was ingestion/consumer.py), pipeline.py (was ../pipeline.py)
  storage/
  features/        engine, process_features, storage_bridge, preprocessing, config
  forecasting/     model, baselines, dataset, evaluation (walk-forward), persistence, training_pipeline
  contention/      model, dataset, evaluation, prediction, processes, persistence, training_pipeline, config
  anomaly/         detector, calibration, persistence, notifications
  lifecycle/       training_data.py, retraining.py, comparison.py, gate.py, deployment.py, active_version.py
  insight/         taxonomy, explainer, diagnosis, text, catalog, recommendations
  evaluation/      ablation.py, report.py, model_comparison.py
  reporting/  replay/
  api/             (unchanged routes) + serves web/
  web/static/      index.html, app.css, js/*.js, vendor/uplot.*
```

- `aiperfopt run` starts the server and opens `http://127.0.0.1:<port>/`
  in the default browser.
- `aiperfopt compare` runs the three-model comparison (was
  `scripts/compare_models.py`).
- No `synthetic.py`, no `frontend/`, no PySide6 or pytest-qt.
- `grep -rnE "\b(N?FR|ADL)-[0-9]|Section [0-9]+|Phase [0-9]+" src` returns nothing.

### Key Discoveries
- Training pipelines already accept pre-built data
  (`scenarios: dict[str, list[SystemMetricSample]]` plus
  `holdout_samples`). The real-usage path builds a one-entry dict
  (`retraining/real_usage_data.py`, `REAL_USAGE_SCENARIO_NAME`). Removing
  synthetic data means making these arguments **required** and deleting
  the `if scenarios is None: generate…` branches
  (`forecasting/training_pipeline.py:449`,
  `contention/training_pipeline.py:194`).
- Per-model **fold evaluation** (`forecasting/evaluation.py`,
  `contention/evaluation.py`) and **artifact save/load**
  (`*/persistence.py`) stay next to their models. The training pipelines
  import them, and `evaluation/ablation.py` imports the training
  pipelines. Moving them into `evaluation/` or `lifecycle/` would create
  package import cycles. (This narrows what was discussed: only
  experiment code moves to `evaluation/`, and only "which model is live"
  code moves to `lifecycle/`.)
- `LiveStateOut` carries everything the Live view needs except chart
  history. History comes from `GET /metrics/system` and `GET /forecasts`
  (forecasts have been persisted since the old Phase 7), so the web chart
  can prefill on page load instead of starting empty.
- `pyproject.toml:75` ships `resources/*.yaml` as package data. The web
  assets need the same treatment, or a non-editable install serves 404s.
- API routes are at the root (`/health`, `/readiness`, …), not under
  `/api`. The dashboard is served at `/` with assets under `/static/`,
  and no route collides. **Routes are not renamed** (tests and the paper
  reference them).
- Node 26 is installed locally, so pure JS formatting helpers can be
  unit-tested with `node --test`, with no npm and no build step.

## What We're NOT Doing

- No changes to model behaviour, features, hyperparameters, thresholds or
  the quality gate's criteria. Any test asserting a model result must
  pass unchanged except where it used synthetic data as input.
- No API route renames or response-schema changes, beyond deleting the
  synthetic-only `RetrainRequest` fields.
- No JS framework, bundler, npm dependency or build step. Plain ES
  modules plus one vendored chart library.
- No native app wrapper (pywebview, Electron). It's a browser tab.
- No Windows work. No artificial load of any kind (CPU, disk, memory,
  network), as before.
- No edits to `paper/` or the PRISM deck. The updated
  `docs/architecture.md` and module table are the source you update them
  from.
- No pushing, PRs or GitHub. Local commits only (standing constraint).

## Implementation Approach

Delete first, then move, then build. Removing synthetic data and dead
symbols shrinks what Phase 3 has to move. Moves happen before docstring
cleanup, so each file is rewritten once, in its final location. The web
UI only depends on HTTP, so it is built against the cleaned-up server,
and Qt is removed last. Each phase ends with the full backend suite
green, `aiperfopt doctor` working, and a local commit.

---

## Phase 1: Checkpoint

### Overview
Commit the untracked work so cleanup starts from a clean tree, and every
later change shows up as a reviewable diff.

### Changes Required
Three local commits on `phase-0-foundations`:
1. `src/aiperfopt/features/preprocessing.py` plus
   `tests/test_features_preprocessing.py`: "Add feature preprocessing (imputation/scaling) used by model comparison"
2. `scripts/compare_models.py`, `scripts/paper_analysis.py`,
   `scripts/paper_figures.py`: "Add three-model comparison and paper analysis scripts"
3. `paper/prism_ieee.tex`, `paper/figures/**`, `.gitignore`
   (`paper_snapshot/`): "Add IEEE paper source and figures".
   Exclude `paper/build.log` and add it to `.gitignore`. Commit
   `prism_ieee.pdf` (it is the deliverable).

### Success Criteria
#### Automated Verification
- [ ] `git status --short` is empty.
- [ ] `pytest --ignore=tests/frontend -q`: 501 passed.

---

## Phase 2: Remove synthetic training

### Overview
Real usage data becomes the only training source.

### Changes Required

#### 1. Product code
- **Delete** `src/aiperfopt/synthetic.py`.
- `forecasting/training_pipeline.py`: delete `generate_training_scenarios`,
  `generate_holdout_scenario`, the `SCENARIO_GENERATORS` /
  `HELD_OUT_SCENARIO_*` imports, and the `scenario_duration_seconds` /
  `interval_seconds` parameters. `train_and_evaluate_all(...)` takes
  `scenarios` and `holdout_samples` as required keyword arguments.
  `dataset_tag` loses its synthetic-derived default and the caller always
  passes it.
- `forecasting/__init__.py`: drop the synthetic re-exports.
- `contention/training_pipeline.py`: same treatment for
  `train_contention_and_anomaly_pipeline`. Keep
  `_interleave_scenario_chunks` only if it is still reachable with a
  single real-usage scenario; otherwise delete it and its test.
- `retraining/pipeline.py`: delete `run_retraining_on_synthetic_data`.
  `RetrainingCandidate.data_source` becomes the constant `"real_usage"`;
  delete the field if nothing reads it except the API echo.
- `anomaly/calibration.py`: delete the `synthetic_features` parameter and
  its concat branch.
- `cli.py`: delete `train --source`, so `train` always uses real data.
  Delete the synthetic branch in `_cmd_train` (`cli.py:~301-330`).
- `api/schemas.py` `RetrainRequest`: delete `data_source`,
  `scenario_duration_seconds`, `interval_seconds`.
- `api/routers/retraining.py`: delete the synthetic branch.
- `frontend/widgets/retraining_panel.py` (still alive until Phase 7):
  delete the data-source combo.
- Comments that only explain synthetic-vs-real differences
  (`features/engine.py:164`, `common/timeutils.py:14`,
  `forecasting/baselines.py:94,109`, `forecasting/model.py:141`,
  `retraining/comparison.py:448`, `config/settings.py:231`): reword
  without the synthetic framing.

#### 2. Tests
- **Delete** `tests/test_synthetic.py`.
- **New** `tests/factories.py`: small, deterministic sample builders
  (numpy RNG with a fixed seed):
  ```python
  def make_samples(n: int, *, start: float = 1_700_000_000.0, interval: float = 2.0,
                   cpu: Callable[[np.ndarray], np.ndarray] | None = None,
                   mem: Callable[[np.ndarray], np.ndarray] | None = None,
                   seed: int = 0) -> list[SystemMetricSample]: ...
  def make_training_data(n_train: int = 300, n_holdout: int = 100, *, with_spikes: bool = True,
                         seed: int = 0) -> tuple[dict[str, list[SystemMetricSample]], list[SystemMetricSample]]:
      """One real-usage-shaped scenario plus a chronologically later holdout,
      with periodic CPU/memory spikes so contention labels have positives."""
  ```
  Default `cpu`/`mem` are a slow sine plus noise. `with_spikes` adds
  short plateaus above the contention labelling thresholds read from
  `ContentionLabelingConfig()` (not hardcoded).
- Port the 13 other test files that referenced synthetic generators
  (`test_forecasting_model`, `test_contention_training_pipeline`,
  `test_retraining_pipeline`, `test_monitoring_process_tracker`,
  `test_skill_gate`, `test_features_engine`, `test_forecasting_evaluation`,
  `test_cli`, `api_conftest`, `test_api_retraining`,
  `test_forecasting_training_pipeline`, `test_forecasting_baselines`,
  `test_anomaly_calibration`, `test_forecasting_dataset`) to
  `tests/factories.py`.
  - Tests whose *subject* was synthetic generation (for example
    `test_held_out_scenario_never_appears_in_walk_forward_training`) are
    rewritten to assert the same property against `make_training_data`
    (holdout timestamps strictly after training), or deleted if the
    property no longer exists.
  - `test_api_retraining`: remove synthetic-source cases. Real-source
    cases seed the test DB via `make_samples`.

### Success Criteria
#### Automated Verification
- [ ] `grep -rn "synthetic" src` returns nothing.
- [ ] `grep -rn "generate_.*scenario\|aiperfopt.synthetic" src tests scripts` returns nothing.
- [ ] `pytest --ignore=tests/frontend -q` is green. Record the new count
  (expected: 501 minus the `test_synthetic.py` tests ± rewritten ones)
  and runtime (expected to drop; synthetic runs generated 900 s scenarios).
- [ ] `aiperfopt train --help` shows no `--source`.
- [ ] `aiperfopt doctor` exits 0 or 2 (not 1).

#### Manual Verification
- [ ] `aiperfopt train` against the real DB completes and prints a candidate summary.

---

## Phase 3: Restructure packages

### Overview
Pure moves and renames. **No logic changes.** Each sub-step is its own
commit, with the suite green after each.

### Changes Required

#### 3a. `lifecycle/` (new)
| From | To |
|---|---|
| `retraining/real_usage_data.py` | `lifecycle/training_data.py` |
| `retraining/pipeline.py` | `lifecycle/retraining.py` |
| `retraining/comparison.py` (candidate-vs-active part: `MetricComparison`, `ContentionComparison`, `RetrainingComparison`, `compare_candidate_to_active` and its `_find_*`/`_regressed_*` helpers) | `lifecycle/comparison.py` |
| `retraining/comparison.py` (gate part: `ForecastGateCriterion`, `ContentionGateCriterion`, `SkillGateResult`, `evaluate_skill_gate`, `_mae_improvement_over_persistence`) | `lifecycle/gate.py` |
| `retraining/deployment.py` + `forecasting/deployment.py` (`load_deployed_forecaster`, `resolve_deployed_forecasters`) | `lifecycle/deployment.py` |
| `retraining/active_version.py` | `lifecycle/active_version.py` |

- `lifecycle/__init__.py` re-exports the names the old
  `retraining/__init__.py` did (minus the synthetic ones), plus
  `load_deployed_forecaster` and `resolve_deployed_forecasters`.
- `forecasting/__init__.py` drops `load_deployed_forecaster`.
- Delete `retraining/` and `forecasting/deployment.py`. No shim modules;
  update every importer (19 src, 8 test, 2 script files reference
  `aiperfopt.retraining`; 5 src and 2 test files reference
  `forecasting.deployment`).
- The per-model `contention/persistence.py`, `forecasting/persistence.py`
  and `anomaly/persistence.py` **stay** (see Key Discoveries).
  `contention/training_pipeline._load_deployed_forecasters`: if it
  duplicates `resolve_deployed_forecasters`, replace it with a call to
  that function. This is the only allowed logic touch, and only if the
  behaviour is identical (verify by reading both).
- Tests: `test_retraining_*` → `test_lifecycle_*`; split
  `test_retraining_comparison.py` / `test_skill_gate.py` to mirror
  `comparison.py` / `gate.py`.

#### 3b. `insight/` (merge `diagnosis/` + `recommendation/`)
| From | To |
|---|---|
| `diagnosis/taxonomy.py` | `insight/taxonomy.py` |
| `diagnosis/explainer.py` | `insight/explainer.py` |
| `diagnosis/text.py` | `insight/text.py` |
| `diagnosis/service.py` | `insight/diagnosis.py` |
| `recommendation/catalog.py` | `insight/catalog.py` |
| `recommendation/service.py` | `insight/recommendations.py` |

- `insight/__init__.py` exports the union of both old `__all__` lists.
- Tests: `test_diagnosis_*` / `test_recommendation_*` → `test_insight_*`.

#### 3c. Fold the producer into `monitoring/`
- `ingestion/consumer.py` → `monitoring/ingestion.py`; delete `ingestion/`.
- `pipeline.py` → `monitoring/pipeline.py`. `monitoring/__init__.py`
  exports `MonitoringPipeline`, `create_monitoring_pipeline` and
  `IngestionConsumer`.
- Tests: `test_ingestion_consumer.py` → `test_monitoring_ingestion.py`,
  `test_pipeline_integration.py` → `test_monitoring_pipeline.py`.

#### 3d. Experiments into `evaluation/` and the CLI
- `scripts/compare_models.py` → `evaluation/model_comparison.py`.
  - Keep the protocol, candidates, `choose_winner` and outputs identical.
  - Replace `sys.path` hacks and `__main__` with
    `run_model_comparison(ctx, out_dir) -> ModelComparisonResult`.
- `cli.py`: new `aiperfopt compare [--out-dir results]` subcommand calling it.
- **New** `tests/test_evaluation_model_comparison.py`: run on a
  `make_training_data` DB with tiny settings. Assert the output files
  exist and a winner is chosen from the three candidates.
  - Mark it `slow` if it takes more than 20 s, and register the `slow`
    marker in `pyproject.toml`.
- `scripts/paper_analysis.py` and `scripts/paper_figures.py` stay as
  scripts (paper-only), with their imports updated.
- `scripts/generate_module_table.py`: update for the new package list.

### Success Criteria
#### Automated Verification
- [ ] After each of 3a–3d: `pytest --ignore=tests/frontend -q` is green with an unchanged test count (3d adds 1+).
- [ ] `python -c "import aiperfopt.cli"` plus the AST reachability walk from Phase 0 research: every module reachable from `aiperfopt.cli`.
- [ ] `grep -rnE "aiperfopt\.(retraining|diagnosis|recommendation|ingestion|pipeline)\b|forecasting\.deployment" src tests scripts` returns nothing.
- [ ] `aiperfopt compare --help` works.

#### Manual Verification
- [ ] `aiperfopt serve` starts, the live loop produces states (`curl /live/state`), and `aiperfopt doctor` shows the same active model as before.

---

## Phase 4: Strip noise

### Overview
Make the code readable to someone without the spec or the plan history.

### Changes Required
1. **Docstrings and comments, all of `src/`:** remove `FR-*`, `NFR-*`,
   `ADL-*`, `R-*`, `Section N`, `Phase N` and `PROJECT_HANDOFF.md`
   references.
   - Rule: keep the *why* in plain words, drop the citation. If a
     paragraph only justified itself by citation, shorten it to one line
     or delete it.
   - Module docstrings: at most ~10 lines unless the module really needs
     more (`api/app.py`'s "nothing starts automatically" note stays,
     shortened).
2. **Delete unused symbols:** the 8 unreferenced exception classes in
   `common/exceptions.py`, `api/dependencies.get_settings` and
   `get_recommendation_catalog`, `api/schemas.RecommendationActionOut`.
   Re-run the unreferenced-symbol scan after Phases 2–3 and delete
   anything new it finds (route handlers and decorators are false
   positives; check by hand).
3. **Docs:**
   - Delete `PROJECT_HANDOFF.md`.
   - Move `docs/plans/2026-08-24-*.md` and `docs/plans/2026-09-12-*.md`
     to `docs/archive/`.
   - Fix test docstrings that cite `PROJECT_HANDOFF.md`
     (e.g. `tests/api_conftest.py:48`).
4. CLI description (`cli.py` `build_parser`): "(Windows and macOS)" → "macOS".

### Success Criteria
#### Automated Verification
- [ ] `grep -rnE "\b(N?FR|ADL)-[0-9]|\bR-[0-9]+\b|Section [0-9]+|Phase [0-9]+|PROJECT_HANDOFF" src tests` returns nothing.
- [ ] `pytest --ignore=tests/frontend -q` is green, same count as end of Phase 3.
- [ ] `wc -l` over `src/` is reported in the commit message (before/after).

---

## Phase 5: Web UI foundation + Live view

### Overview
A dashboard served by FastAPI at `/`, dark-first, with the Live view at
full fidelity. Qt keeps working in parallel.

### Changes Required

#### 1. Serving
- **New** `src/aiperfopt/web/static/` (package data):
  - `index.html`, `app.css`
  - `js/app.js` (shell, router, connection)
  - `js/api.js` (fetch and WebSocket wrapper with reconnect/backoff)
  - `js/format.js` (pure helpers, ported from `frontend/formatting.py`
    incl. `humanize_category`, `confidence_tier`, `readiness_banner_text`)
  - `js/charts.js` (uPlot wrappers)
  - `js/views/live.js`
  - `vendor/uPlot.iife.min.js` and `vendor/uPlot.min.css`: uPlot 1.6.x,
    MIT, about 50 KB, vendored so the demo works offline. **Downloading
    it is a one-time step that needs your OK when we get there.**
- `api/app.py`:
  ```python
  _WEB_DIR = Path(__file__).resolve().parent.parent / "web" / "static"
  app.mount("/static", StaticFiles(directory=_WEB_DIR), name="static")

  @app.get("/", include_in_schema=False)
  def index() -> FileResponse:
      return FileResponse(_WEB_DIR / "index.html")
  ```
  Mount this after the routers.
- `pyproject.toml` package-data:
  `aiperfopt = ["resources/*.yaml", "web/static/*", "web/static/**/*"]`.

#### 2. Design system (`app.css`)
- Tokens on `:root`:
  - Surfaces `--bg`, `--surface`, `--surface-2`, `--border`.
  - Text `--text`, `--text-muted`.
  - Accent `--accent`.
  - Semantic `--ok`, `--warn`, `--danger`, `--replay`.
  - Series colors `--series-cpu`, `--series-mem`, `--series-contention`,
    picked for contrast on both themes.
  - Spacing on a 4 px scale (`--s1`…`--s8`), radius `--r`, and the font
    stack `-apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui`
    with `font-variant-numeric: tabular-nums` on every number.
- Dark is the default. Light is applied under
  `@media (prefers-color-scheme: light)`.
- Charts read their colors from CSS variables at render time, so there is
  no white chart box in a dark window.
- Responsive: the sidebar collapses into a top bar under 900 px.

#### 3. App shell (`index.html`, `app.js`)
```
┌──────────┬──────────────────────────────────────────────────────┐
│ PRISM    │  ● Live · ready        [Monitoring ●] [Predictions ●] │  ← status bar
│          ├──────────────────────────────────────────────────────┤
│ Live     │                                                      │
│ Accuracy │                    view content                      │
│ Reports  │                                                      │
│ Model    │                                                      │
└──────────┴──────────────────────────────────────────────────────┘
```
- Hash routing: `#/live`, `#/accuracy`, `#/reports`, `#/model`.
- **The General/Developer mode toggle is dropped.** Technical detail goes
  in the Model view instead of a mode switch.
- Status bar:
  - A connection dot (WS open, reconnecting, or down).
  - A readiness pill from `GET /readiness`, polled every 3 s:
    `no_data` / `collecting` (with "N s of M s collected") /
    `needs_training` / `ready` / `replay`. Replay gets its own violet
    style and a persistent "REPLAY: recorded session" label.
  - Monitoring and Predictions toggles (`/monitoring/*`, `/live/*`) that
    reflect real status on load.

#### 4. Live view (`views/live.js`)
- **Tiles row:** CPU %, Memory %, Swap %, and Contention risk (probability
  plus a "threshold X%" caption, colored when `triggered`). Each tile has
  a 60-point sparkline.
- **Two stacked charts with a shared time axis** (replacing the one
  chart with six legend entries): CPU and Memory. Each shows:
  - actual as a solid line
  - forecast as a dashed continuation to `target_timestamp`, with the
    confidence band shaded
  - a "now" marker
  - y-axis fixed to 0–100 %
  - human labels ("CPU", "in 30 s") taken from `horizon_seconds`
- **Contention strip:** probability line plus a dashed threshold line,
  with triggered regions shaded. Same time axis.
- **Prefill on load:** `GET /metrics/system?start=now-15min` and
  `GET /forecasts?start=now-15min`, then append from `WS /ws/live`.
  Buffer capped at 15 min.
- **"What's happening" card:** diagnosis title (`humanize_category`),
  plain-language explanation and confidence chip. Shows an anomaly notice
  when `anomaly_notification`. When nothing is wrong it says
  "All clear" in muted text, with no empty box.
- **Recommendations:** up to 3 action cards (title plus description), and
  a "Top contributing processes" list from
  `contention.contributing_processes`.
- **Errors:** `state.error` shows as a dismissible danger banner.

#### 5. Tests
- **New** `tests/test_web_static.py`:
  - `GET /` returns 200 `text/html` containing `<div id="app">`.
  - Every `<script src>` / `<link href>` in `index.html` and every
    `import` in `js/**/*.js` resolves to a 200 under `/static/`.
  - The `/static/` mount doesn't shadow `/health`.
- **New** `src/aiperfopt/web/static/js/format.test.mjs`, run with
  `node --test`, covering the helpers ported from `formatting.py` (mirror
  `tests/frontend/test_formatting.py` cases).
  - `pytest` runs it via
    `tests/test_web_format_js.py::test_node_format_tests`, which calls
    `subprocess.run(["node", "--test", ...])` and is skipped when `node`
    is absent.

### Success Criteria
#### Automated Verification
- [ ] `pytest --ignore=tests/frontend -q` is green, including the new web tests.
- [ ] `node --test src/aiperfopt/web/static/js/` passes.
- [ ] `pip wheel . -w /tmp/w && unzip -l /tmp/w/aiperfopt-*.whl | grep web/static/index.html` (the assets ship in the wheel).

#### Manual Verification (built-in browser pane, screenshots shared with you)
- [ ] `aiperfopt serve` then open `/`: the Live view renders in dark and light, with no white boxes and readable status text.
- [ ] Reload mid-session: the charts are prefilled, not empty.
- [ ] Stop and restart the server: the connection dot goes to reconnecting, then recovers without a page reload.
- [ ] Replay DB (`aiperfopt replay …` plus `serve --no-monitor`): the violet REPLAY pill is shown.
- [ ] 375 px width: no horizontal scroll, and the sidebar collapses.

---

## Phase 6: Web UI, remaining views

### Changes Required
- **Accuracy view** (`views/accuracy.js`), replacing Predicted vs. Actual:
  - Controls: metric (from `GET /reporting/metrics`, shown as human
    labels), horizon (short/medium as "in N s"), lookback
    (15 min / 1 h / 6 h / 24 h segmented control).
  - Chart: actual vs. predicted, in the same visual language as Live.
  - Under the chart: MAE, and "points compared" stats computed from the
    returned rows.
  - Empty state explains why: "No forecasts persisted in this window,
    start Predictions."
- **Reports view** (`views/reports.js`):
  - Period control (daily/weekly) driving `GET /reporting/summary`.
  - Sections as cards: system context, prediction accuracy, contention
    events, diagnoses by category (horizontal bars), recommendations
    issued.
  - Field mapping is taken by reading `reporting/summary.py` and the
    existing `frontend/widgets/reporting_tab.py`, so nothing it shows
    today is lost.
- **Model view** (`views/model.js`), covering the old Developer panel and
  Retraining:
  - Active model version and latest deployment (`/deployments/latest`).
  - Feature attribution for the latest diagnosis
    (`diagnosis.forecast_explanations`) as a bar list.
  - Contention internals: contributing metrics and processes, anomaly
    score.
  - Raw feature-vector table, collapsible and searchable.
  - **Retraining panel:**
    - Inputs: version name, optional metrics. "Run retraining" calls
      `POST /retraining/run` with a progress state.
    - Shows the candidate-vs-active comparison table and the
      quality-gate verdict, with blocked criteria in danger color.
    - "Confirm deployment" is disabled while the gate blocks. It needs a
      second click: "Confirm — this replaces vX".
    - "Reject" and "Rollback to last known good" sit beside it.
  - All actions are disabled while a request is in flight.
- Parity checklist: walk every widget in `src/aiperfopt/frontend/widgets/`
  and tick each field it shows against the web view that now shows it.
  Keep the list in the PR/commit message.

### Success Criteria
#### Automated Verification
- [ ] `pytest --ignore=tests/frontend -q` is green. `test_web_static.py` covers the new view modules (import-resolution check).
- [ ] `node --test` passes (new helpers for labels and stats have cases).

#### Manual Verification
- [ ] Every item on the parity checklist is visible in the web UI.
- [ ] Retraining end to end on the real DB: run, see the comparison, the gate blocks or allows correctly, and confirm/rollback write a deployment (visible in `/deployments`).
- [ ] Screenshots of all four views (dark) replace `paper/figures/live_dashboard.png`-style evidence for the deck. Saved to `docs/screenshots/`.

---

## Phase 7: Delete Qt, wire everything up

### Changes Required
- **Delete** `src/aiperfopt/frontend/` and `tests/frontend/`.
- `cli.py`:
  - `run` starts the API server in-process (`aiperfopt.api.server.main`,
    `auto_start=True`), waits for `/health` on a helper thread, then
    calls `webbrowser.open(base_url)` and prints the URL.
  - Ctrl-C shuts down cleanly (uvicorn handles the signal; lifespan stops
    the loops).
  - Delete `ui` and `_cmd_ui`.
  - `run`'s subprocess and Qt-main-thread comment goes away.
- `pyproject.toml`:
  - Remove `pyside6` and `pytest-qt`.
  - Add `fastapi`'s `StaticFiles` dependency `aiofiles` only if the
    installed Starlette needs it (check; recent Starlette does not).
  - Drop the Qt-specific `timeout` comment.
  - Regenerate `requirements.lock`.
- `.github/workflows/ci.yml`:
  - Delete the frontend job.
  - Add `actions/setup-node` to the backend job so the node tests run in
    CI rather than being skipped.
  - Remove `--ignore=tests/frontend`.
- Docs:
  - `README.md`: run instructions (`aiperfopt run` opens the browser),
    screenshots, a command table without `ui` or `--source`, with
    `compare`.
  - `docs/architecture.md`: Mermaid updated with the new package names,
    "Frontend: web dashboard (served by FastAPI)" and the `lifecycle` /
    `insight` boxes.
  - `docs/phase10-module-table.md`: regenerated via
    `scripts/generate_module_table.py`, then renamed to
    `docs/module-table.md`.
  - `docs/phase10-demo-script.md` → `docs/demo-script.md`, with steps
    updated for the browser.
  - **New** `docs/paper-deck-changes.md`: a short list of what changed
    that the paper and deck still describe (PySide6 → web, module
    renames, synthetic training removed) for you to apply.

### Success Criteria
#### Automated Verification
- [ ] `pytest -q` (no ignores) is green.
- [ ] `grep -rniE "pyside|qt|frontend" src pyproject.toml .github` returns nothing, except an intentional mention in history docs.
- [ ] `pip install . && aiperfopt --help` in a fresh venv works without PySide6.
- [ ] `aiperfopt run` prints the URL, and `/` returns 200 (curl).

#### Manual Verification
- [ ] `aiperfopt run` opens the browser on the Live view with live data. Ctrl-C exits within a few seconds, with no orphaned process (`pgrep -fl aiperfopt` is empty).
- [ ] Run the demo script (`docs/demo-script.md`) once, end to end, with replay.

---

## Testing Strategy

### Unit
- `tests/factories.py` replaces synthetic data. Keep it small and
  deterministic, with thresholds read from config.
- The JS formatting helpers get `node --test` cases mirroring the old
  Python formatting tests.
- `test_web_static.py` checks HTML, assets, import resolution and route
  non-collision.

### Regression guard for "no behaviour change"
- Before Phase 2, save `aiperfopt doctor` output and
  `curl /live/state | jq 'del(.timestamp)'` shape. After Phases 3 and 4,
  the same keys are present and the same active model loads.
- Phase 3 commits are moves only, so `git diff -M --stat` should show
  mostly renames.

### Manual (each UI phase)
1. `aiperfopt serve`, then open `/` in the built-in browser pane, dark
   and light.
2. Reload mid-stream (prefill), kill the server (reconnect), replay DB
   (REPLAY pill), 375 px width.

## References
- Prior plan (completed): `docs/plans/2026-09-12-presentable-predictive-monitor.md` → `docs/archive/` in Phase 4
- Current UI evidence: `paper/figures/live_dashboard.png`, `paper/figures/predicted_vs_actual.png`
- Live payload: `src/aiperfopt/api/schemas.py` `LiveStateOut`
- Training-data entry: `src/aiperfopt/retraining/real_usage_data.py` `build_real_usage_training_data`
- Pipelines' pre-built-data path: `src/aiperfopt/forecasting/training_pipeline.py:414-450`, `src/aiperfopt/contention/training_pipeline.py:164-195`
