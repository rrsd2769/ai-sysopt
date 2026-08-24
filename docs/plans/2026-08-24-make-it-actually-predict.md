# Make It Actually Predict — Repair Plan

**Date:** 2026-08-24
**Status:** proposed (rev 2 — Windows added as a first-class target)
**Basis:** measured runtime behaviour on a real machine. `PROJECT_SPEC.md` and `PROJECT_HANDOFF.md` were deliberately ignored.

---

## Overview

The plumbing is sound and the test suite is green, but **no model in the system has measurable predictive skill**, a single train/serve bug makes most of the product unreachable, and the project is macOS-only while it is required to run on Windows. This plan fixes the intelligence, makes "predictive" a falsifiable claim enforced by a hard gate, makes the UI prove the prediction, and makes Windows a real target rather than an aspiration.

## Decisions Taken

| Question | Decision |
|---|---|
| Success bar | Genuinely predictive — real skill, proven in the UI |
| Training data | Stress harness driving real load on real hardware |
| Scope | Fix the blocker, keep all phases alive |
| Model goal | Make LightGBM genuinely win |
| Quality gate | **Hard blocking** — refuses deployment, fails CI |
| Stress limits | **CPU + disk only, capped.** No memory pressure. |
| Frontend | Keep PySide6, add real charts |
| Time budget | Open-ended; foundations first, then iterate on measured quality |
| Test suite | Accepted that some of the 355 tests are worse than nothing — classify and mark them |
| Docstrings | Accepted: trade docstring volume for tests asserting the same claims |
| Per-phase E2E | **Mandatory.** Every phase ends with a full-stack run before the next begins |
| Thresholds | Fit from data, not chosen by argument |
| **Platform** | **Windows and macOS both. Windows is a real target.** |

---

## Current State Analysis

Environment: fresh venv, Python 3.12.14, `pip install -e ".[dev]"` succeeds. All findings measured, not inferred.

### What works

| Area | Evidence |
|---|---|
| Backend tests | `355 passed` in 121s (`pytest --ignore=tests/frontend`) |
| Monitoring | 14 real samples in 20s at 1.5s interval, all fields populated |
| Training | 38s on synthetic data, 1720 contention rows |
| Deployment | Artifacts, active-version pointer, deployment record all written |
| Live loop | 200s / 39 cycles, zero errors |
| **Portability of collection** | **No `sys.platform`, no `fork`, no signal handling anywhere in `src/`.** Pure psutil with per-source degradation. |

This is not a broken-wiring problem, and it is not a from-scratch rewrite.

### Findings

**F1 — The anomaly detector returns the maximum score on every real sample.**
39/39 live feature vectors scored exactly `1.0`, including the 8 with zero missing features. Causes:
- Train/serve imputation skew: `fit()` fills NaN with column means (`anomaly/detector.py:83`); `score()` and `score_batch()` fill with `0.0` (`:100`, `:111`). A live `cpu_percent_lag_30s` of `0.0` never occurs in training. 24 of 88 features were NaN on typical live rows.
- Deeper: synthetic-only training makes real data out-of-distribution by construction, which is why even zero-NaN rows saturate.

**F2 — F1 makes most of the product unreachable.**
`resolve_diagnosis_category` tests anomaly **first** (`diagnosis/taxonomy.py:228`). Score `1.0` ≥ threshold `0.7` always returns `UNUSUAL_PATTERN_ANOMALY`, which is non-actionable. Measured: 39/39 cycles that category, `recommendations` table 0 rows. **All 7 actionable categories, the SHAP explainer, and the entire recommendation engine are dead code at runtime.**

**F3 — Contention never triggers, and the classifier has no skill.**
Dampening multiplies probability by `1 − 0.3×1.0 = 0.7` permanently (`contention/prediction.py:59`). Live probabilities 0.10–0.20 against a 0.75 threshold. Training reported walk-forward `precision 0.0125, f1 0.024` and **holdout `f1 0.0`** — deployed with `regressed: False`. No quality gate exists; F1 of 0.0 passes all 355 tests.

**F4 — LightGBM loses to naive baselines on 4/4 metric-horizons.**
Selected: `moving_average`, `moving_average`, `exponential_smoothing`, `exponential_smoothing`. Directional accuracy 0.41–0.51 (coin flip). CPU medium-horizon MAE 25.4 points on a 0–100 scale. Structural cause: `feature_cols` excludes `current_value` (`forecasting/training_pipeline.py:265`, duplicated as `_NON_FEATURE_COLUMNS` in `forecasting/evaluation.py:32`); the feature matrix holds only lags, rolling stats and rate-of-change. **The model never sees the current value of the metric it predicts, while the persistence baseline it is benchmarked against receives it directly.** Compounding: the target is an absolute future level, which trees cannot extrapolate. This is the identical failure mode already diagnosed for counter metrics in `forecasting/dataset.py`'s docstring and never applied to gauges.

**F5 — Forecasts are computed and discarded.**
`forecast_predictions`: **0 rows** after a full live run. `live_state._tick` computes forecasts inside `build_live_contention_input_row` and never persists them. `ForecastPredictionRecord` and `insert_forecast_predictions_batch` (`storage/writer.py:155`) already exist and are never called. Consequence: `/reporting/predicted-vs-actual`, the Predicted vs. Actual tab, and the report's accuracy section are permanently empty.

**F6 — The full test suite hard-crashes the interpreter.**
`pytest tests/frontend` → **exit 139, `Fatal Python error: Segmentation fault`** after ~45 tests, at `tests/frontend/test_main_window.py:74`. Crash dump shows monitoring, ingestion and uvicorn threads all still alive. Contributors: `server_fixture.running_backend` joins with `timeout=5.0` and continues regardless (`tests/frontend/server_fixture.py:76`), leaking live threads into the next test; `LiveWebSocketClient._socket = QWebSocket()` is created **unparented** (`frontend/live_client.py:55`) while `QTimer(self)` is parented, leaving destruction order undefined; `app._broadcast` calls `asyncio.run_coroutine_threadsafe` on a loop that may be closing (`api/app.py:87`).

**F7 — No entry point, no cold start.**
No `console_scripts`, no `__main__.py`. Fresh install → server starts → live loop returns early at `api/live_state.py:157` forever, because `get_active_model_version()` is `None` until someone hand-writes a training script.

**F8 — Paths are relative to CWD.**
`data/aiperfopt.db`, `models`, `logs` resolve against the working directory. Launching from another directory silently creates a second empty database. Reproduced in minutes.

**F9 — The dashboard has no charts.**
`DashboardWidget` is three number tiles and text labels. For a *predictive* monitor there is nowhere to see a trend or a forecast. QtCharts is installed and used in exactly one tab (`frontend/widgets/predicted_vs_actual.py:14`).

**F10 — Process-metric volume.**
26,451 `raw_process_metrics` rows per ~5 minutes ≈ 7.6M/day; at 72h retention, ~23M rows in SQLite.

**F11 — The test suite creates false confidence.**
355 tests pass while every model has zero skill. The suite asserts that functions are called, records written, threads started and stopped. It asserts nothing about answer quality. This is why Phases 5–11 were built on a broken Phase 4 without anyone noticing.

**F12 — Docstrings describe intent, not behaviour, and read as verification.**
`forecasting/dataset.py` argues at length why counter metrics were "structurally set up to fail" — and never applies the identical argument to gauges (F4). `anomaly/detector.py:67` documents mean-imputation as a "deliberate, narrow exception" while the inference path does something else (F1). Prose in place of assertions.

### Windows findings

The collection layer is portable; the *semantics* are not. Four issues, in severity order.

**W1 — `load_avg_*` is silently wrong on Windows, and it is a model input.**
`psutil.getloadavg()` is **emulated** on Windows: it does not raise, returns `(0.0, 0.0, 0.0)` for the first ~5 minutes of uptime, and thereafter a psutil-simulated value with different semantics from a Unix load average. `_safe_source` (`monitoring/collectors.py:84`) therefore cannot null it out — nothing raises. Meanwhile `load_avg_1` is:
- a `gauge_metric` in `FeatureEngineeringConfig` (lags, rolling stats and rate-of-change computed on it),
- a forecast target metric,
- part of `_CATEGORY_METRIC_NAMES` for both CPU diagnosis categories,
- in `VALID_METRIC_NAMES` for predicted-vs-actual reporting.

So on Windows the model trains on zeros-then-simulation for a core feature, and the CPU diagnosis explains a quantity that does not mean what it says. **This is the single most damaging Windows defect and it fails silently.**

**W2 — User-facing advice is macOS-only.**
`config/recommendations.yaml` tells the user to use "Activity Monitor's CPU tab" (`:77`), "macOS's `nice`/Activity Monitor priority controls" (`:47`), and names "Spotlight indexing, and Time Machine backups" (`:143`). `diagnosis/text.py:112` and `:118` explain memory behaviour as "macOS starts moving less-used data out to disk". On Windows the equivalents are Task Manager, priority classes, Windows Search indexing, File History/OneDrive, and the page file. A Windows user receives confidently-worded wrong instructions.

**W3 — `priority` mixes two incompatible scales.**
`process_tracker.py:128` stores `proc.nice()` directly. On POSIX that is −20…19; on Windows it is a priority-class constant (32, 128, 256…). One column, two non-comparable scales — meaningless in any cross-platform aggregate or display.

**W4 — The stress harness (Phase 5) must be spawn-safe from the start.**
Windows `multiprocessing` uses `spawn`, not `fork`: worker targets must be importable top-level functions with picklable arguments and a `__main__` guard. `SIGKILL` does not exist; cleanup must go through `psutil.Process.terminate()`/`.kill()` rather than raw signals. Designing the harness fork-first would require rewriting it.

Lesser items folded into the phases: app-data directory should be `%LOCALAPPDATA%` on Windows; process names carry `.exe` suffixes, which affects the privacy exclusion list and diagnosis text; every "macOS system health monitor" string in `src/`, `README.md` and `api/app.py:105` is now false.

### Verification limitation, stated plainly

**I cannot verify anything on Windows from this machine.** I can write portable code, remove the platform assumptions above, and add a Windows CI matrix — but the Windows leg of every phase's success criteria must be run by you or by CI. Every phase below therefore separates **macOS-verified** from **Windows-verified (requires CI or your machine)**.

This also means CI is now load-bearing, and **the project is not currently a git repository** — `git init` plus a remote is a prerequisite for the Windows leg.

---

## Desired End State

- One command installs, starts, trains, and shows a working UI **on both Windows and macOS**.
- The metric set is explicit about what each platform can actually measure; nothing silently substitutes a simulated value for a real one.
- On a real machine, the anomaly detector flags ordinary usage **rarely**.
- LightGBM **beats naive persistence** on CPU and memory short-horizon forecasts, on a holdout it never saw.
- A **hard gate** refuses to deploy any model failing to beat its baseline — at runtime and in CI, on both platforms.
- Thresholds are fitted from data, with the curve recorded.
- The UI shows a live trend with the forecast projected ahead, and a predicted-vs-actual overlay proving the call preceded the event.
- Every test is labelled behavioural or plumbing, and no surface displays a number the evaluation does not support.

## What We're NOT Doing

- Not rewriting the frontend as a web app. PySide6 stays.
- Not inducing memory pressure or swap thrash.
- Not deleting the recommendation engine, retraining UI, or reporting — they get unblocked.
- Not adding deep learning or a second model family. LightGBM gets a fair fight; that is the experiment.
- Not adding per-prediction quantile calibration. Confidence stays a validation-derived scalar.
- Not supporting Linux. Windows and macOS only — but the capability model in Phase 1 makes Linux cheap later.
- Not chasing the README's "402 passed". The real number is whatever passes after F6.
- Not touching `PROJECT_SPEC.md` / `PROJECT_HANDOFF.md`.

## Definition of Done (every phase)

A phase is not complete until all four hold:

1. Its automated criteria pass on macOS.
2. Its automated criteria pass on Windows (CI or your machine).
3. **A full end-to-end run has been performed** — `aiperfopt doctor` → `collect` → `train` → `run` → observe the UI — not just unit tests. This is the discipline whose absence let 11 phases be built on a broken foundation.
4. Any docstring claim the phase touched is either asserted by a test or deleted.

---

## Phase Order and Rationale

Dependency-driven, not importance-driven.

| # | Phase | Why here |
|---|---|---|
| 0 | Runnable, reproducible, cross-platform foundations | Nothing is verifiable without it |
| 1 | **Platform capability model** | Settles the metric set. Doing model work first means redoing it |
| 2 | Test-suite honesty audit | Makes the suite trustworthy *before* we rely on it to prove fixes |
| 3 | **Unblock the anomaly detector** | Highest single leverage — revives 8 dead surfaces |
| 4 | Give the forecaster a fair fight | Depends on the settled metric set |
| 5 | Stress harness (spawn-safe) | Needs the platform work; feeds Phase 6 |
| 6 | **Quality gate + fitted thresholds** | Needs real data to fit against |
| 7 | Persist forecasts | Small; makes the claim demonstrable |
| 8 | UI charts, segfault, platform text | Needs something true to display |
| 9 | Storage & overhead | Independent; deferred deliberately |
| 10 | Revive dead surfaces honestly | Verifies Phase 3 actually paid off |

---

## Phase 0 — Runnable, reproducible, cross-platform foundations

### Changes Required

**1. Pin the environment.** `pyproject.toml`: `requires-python = ">=3.12,<3.13"`; upper bounds on the numeric stack (`pandas<3`, `numpy<3`, `scikit-learn<2` — the current install pulled pandas 3.0.5 / numpy 2.5.2, bleeding edge and a plausible F6 contributor); add `pytest-timeout` so a hung Qt test fails instead of blocking; generate `requirements.lock`.

**2. One app data root, absolute and platform-correct.** `config/settings.py`: introduce `AIPERFOPT_HOME`, defaulting to `%LOCALAPPDATA%\aiperfopt` on Windows and `~/Library/Application Support/aiperfopt` on macOS. Every relative path in `config/default.yaml` (`storage.database_path`, `forecasting.models_dir`, `retraining.active_version_file`, `logging.log_dir`) resolves against it through a validator returning an absolute `Path`. User-supplied absolute paths pass through. Fixes F8.

**3. A real CLI.** New `src/aiperfopt/cli.py` and `src/aiperfopt/__main__.py`; `[project.scripts] aiperfopt = "aiperfopt.cli:main"`. Subcommands: `doctor` (environment, platform, paths, DB, model version, readiness), `serve`, `ui`, `collect --minutes N`, `train [--source synthetic|real] [--deploy]`, `stress <scenario>` (Phase 5), `run` (serve + wait for `/health` + launch UI — the one command). Fixes F7.

**4. Readiness, surfaced.** `api/routers/system.py`: `GET /readiness` returning a discriminated state — `needs_data` (seconds collected vs. required), `needs_training`, `model_below_gate`, `ready`. Consumed by Phase 8's honest empty states.

**5. Repo and CI.** `git init`, then a GitHub Actions matrix over `windows-latest` and `macos-latest` running the backend suite, and the frontend suite headless. Without this the Windows leg of every later phase is unverifiable.

### Success Criteria

**Automated (macOS):**
- [ ] Clean 3.12 venv install from `requirements.lock` succeeds
- [ ] `pytest --ignore=tests/frontend -q` → 355+ passed
- [ ] New test: all settings paths resolve absolute under a `tmp_path` `AIPERFOPT_HOME`
- [ ] New test: launching from two different CWDs reaches the *same* database file
- [ ] New test: the default data root matches the platform convention on the current OS
- [ ] `aiperfopt doctor` exits 0 when healthy, non-zero with a named cause otherwise

**Automated (Windows — CI):**
- [ ] Install succeeds on `windows-latest`
- [ ] `pytest --ignore=tests/frontend -q` → same pass count as macOS
- [ ] Data root resolves under `%LOCALAPPDATA%`

**Manual:**
- [ ] `aiperfopt run` from any directory opens a window connected to a live backend
- [ ] `aiperfopt doctor` on a fresh install says "no model trained — run `aiperfopt train`"

**Pause for confirmation.**

---

## Phase 1 — Platform capability model

### Overview
Fixes W1, W2, W3. This must precede all model work: it determines which metrics exist, and therefore what every feature, model and diagnosis is built on. Doing Phase 4 first would mean redoing it.

### Changes Required

**1. Declare capabilities explicitly.** New `src/aiperfopt/platform.py`: a `MetricCapability` per metric with one of three states — `native` (the OS measures it directly), `emulated` (psutil simulates it; present but not comparable), `unavailable`. Resolved once at startup from `sys.platform` plus probing. Windows marks `load_avg_*` as **`emulated`**.

**2. Emulated metrics never silently enter a model.** `features/config.py` and `forecasting/training_pipeline.py`: derive `gauge_metrics`, `counter_metrics` and the target-metric list from the capability model rather than from hardcoded tuples. An `emulated` metric is excluded from features, targets and diagnosis attribution by default, and is recorded on the model artifact so a macOS-trained model is never loaded against a Windows metric set (or vice versa) without a loud mismatch error. This is the direct fix for W1.

**3. Platform-correct user-facing text.** `config/recommendations.yaml` gains per-platform action variants keyed by platform (`windows` / `darwin`) wherever the advice is OS-specific — the Activity Monitor references at `:47` and `:77`, and the Spotlight/Time Machine list at `:143`. `diagnosis/text.py:112`/`:118` take the platform name and the correct paging vocabulary ("page file" vs "swap"). The catalog loader must error on a category missing a variant for the running platform, matching the fail-loudly discipline the file's own comments already claim. Fixes W2.

**4. Normalize `priority`.** `monitoring/process_tracker.py:128`: store a platform-neutral ordinal (e.g. `-2…+2` from "idle" to "realtime") alongside the raw value, mapping POSIX nice and Windows priority classes onto one comparable scale. Fixes W3.

**5. Retire the macOS-only claims.** `src/aiperfopt/__init__.py`, `api/app.py:105`, `storage/schema.py:119`, `README.md`.

### Success Criteria

**Automated (macOS):**
- [ ] New test: capability resolution returns `native` for `load_avg_1` on darwin
- [ ] New test: an `emulated` metric is absent from the feature matrix, the target list, and diagnosis attribution
- [ ] New test: loading a model artifact whose capability fingerprint differs from the running platform raises
- [ ] New test: every actionable category has a text variant for **both** platforms, and the loader raises when one is missing
- [ ] New test: the priority ordinal is identical for equivalent POSIX and Windows priority inputs
- [ ] No occurrence of "macOS" remains in `src/` outside a platform-conditional branch

**Automated (Windows — CI):**
- [ ] Capability resolution returns `emulated` for `load_avg_1`
- [ ] The trained feature set on Windows **excludes** `load_avg_*`
- [ ] Recommendation text renders Windows variants (Task Manager, not Activity Monitor)
- [ ] `proc.nice()` maps to the expected ordinal

**Manual:**
- [ ] `aiperfopt doctor` prints a capability table showing what this OS can and cannot measure
- [ ] A Windows run's recommendations name Windows tools

**Pause for confirmation.**

---

## Phase 2 — Test-suite honesty audit

### Overview
Fixes F11 and F12. The suite currently manufactures confidence. Before relying on it to prove any later fix, make it say what it actually checks.

### Changes Required

**1. Classify every test.** Mark each of the 355 with `@pytest.mark.behavioural` (asserts an answer is correct) or `@pytest.mark.plumbing` (asserts a call happened, a record was written, a thread started). Add `--strict-markers`. Publish the ratio in the README. The expectation, based on F3, is that behavioural coverage of model quality is near zero.

**2. Convert docstring claims into assertions or delete them.** Working through the highest-risk prose first: `forecasting/dataset.py`'s counter-metric argument, `anomaly/detector.py:67`'s imputation claim, `features/engine.py`'s no-train/serve-skew claim, `contention/dataset.py`'s FR-6.1 boundary claim. Each becomes a test or goes away. This is the accepted trade: less prose, more proof.

**3. Add the missing behavioural tests as failing tests now.** Each of the following is written in this phase and **expected to fail** until its phase lands, with `xfail(strict=True)` naming the finding — so progress is mechanically visible rather than asserted:
- anomaly score on training data is low (F1)
- `score()` and `score_batch()` agree (F1)
- a CPU category is reachable during a high-anomaly window (F2)
- LightGBM beats persistence on cpu/short (F4)
- a live tick persists forecast rows (F5)
- the frontend suite exits 0 (F6)

### Success Criteria

**Automated (both platforms):**
- [ ] Every test carries exactly one of the two markers; `--strict-markers` passes
- [ ] `pytest -m behavioural` and `pytest -m plumbing` both run cleanly
- [ ] The six `xfail(strict=True)` tests exist and fail for the documented reason
- [ ] No docstring in the four named modules claims behaviour no test asserts

**Manual:**
- [ ] README states the behavioural/plumbing ratio honestly

**Pause for confirmation.**

---

## Phase 3 — Unblock the anomaly detector (the critical fix)

### Overview
Fixes F1 and F2. Highest leverage in the document: revives the recommendation engine, the SHAP explainer, and all 7 actionable categories.

### Changes Required

**1. One imputation, used everywhere.** `anomaly/detector.py`: persist the training column means on the artifact; add one private `_prepare(X)` that reindexes to `feature_columns` and fills from those means, falling back to `0.0` only for columns entirely NaN in training. Call it from `fit`, `score` and `score_batch`. Delete the two divergent `fillna(0.0)` sites. Bump the artifact schema; `load_anomaly_detector` must reject an artifact lacking the means rather than loading one that behaves like today.

**2. Calibrate against real normality.** The detector must be fit on data from the machine it scores. Add a calibration step over **real collected feature vectors** (passive collection — safe, no stress required). Synthetic may contribute, but real normal data must be present or the detector refuses to fit for live use.

**3. Stop the anomaly branch short-circuiting diagnosis.** `diagnosis/taxonomy.py`: evaluate contention triggering and resource dominance **first**; `UNUSUAL_PATTERN_ANOMALY` becomes the fallback when nothing else resolves, and a high anomaly score becomes a flag on the record rather than a category pre-empting all others. Preserves the category and its non-actionable status while making the other 7 reachable during a genuinely anomalous window.

### Success Criteria

**Automated (both platforms):**
- [ ] `score(row)` and `score_batch(row.to_frame().T)` return **identical** values — the Phase 2 xfail flips to pass
- [ ] Scoring the training matrix yields mean score < 0.3 — xfail flips to pass
- [ ] On held-out **real** normal data, fewer than 10% of rows exceed the significance threshold
- [ ] High anomaly + triggered CPU contention resolves to a CPU category — xfail flips to pass
- [ ] Each of the 7 actionable categories is reachable from a constructed contention record
- [ ] Loading a pre-Phase-3 artifact raises
- [ ] Backend suite still green

**Manual:**
- [ ] `aiperfopt collect --minutes 10` then a live run: the amber "unusual behaviour" banner is **absent** during ordinary use
- [ ] Contention probability is no longer uniformly cut by 30%
- [ ] **Full E2E run** per the Definition of Done

**Pause for confirmation.**

---

## Phase 4 — Give the forecaster a fair fight

### Overview
Fixes F4. Remove the structural handicap, then let LightGBM earn the win on the Phase 1 metric set.

### Changes Required

**1. One source of truth for non-feature columns.** `_NON_FEATURE_COLUMNS` (`forecasting/evaluation.py:32`) and the inline literal at `forecasting/training_pipeline.py:265` are two copies of one decision. Collapse to a single exported constant — changing one without the other silently produces the train/serve column mismatch this phase exists to remove.

**2. Give the model the current value.** Remove `current_value` from the excluded set. The live path **already** provides it: `build_live_contention_input_row` sets `predict_row["current_value"]` (`contention/dataset.py:196`) for the baselines. Verify the third consumer too — `diagnosis/explainer.py` builds rows for SHAP and must carry the same column, or attribution explains a different vector than the model saw.

**3. Predict the delta, not the level.** `forecasting/dataset.py`: add `target_delta = target − current_value`, train on it, reconstruct `predicted = current_value + predicted_delta` in `MetricForecaster.predict`. Trees cannot extrapolate an absolute level; they can learn a residual against persistence. Keep **all evaluation in absolute units** so MAE/RMSE remain comparable to the baselines and to today's numbers.

**4. Retune.** `forecasting/model.py`: `num_boost_round=80` with no early stopping and `min_data_in_leaf=5` is a guess. Add early stopping against the final walk-forward fold, raise the ceiling, record the chosen iteration count on the artifact.

**5. Add features, with a causality assertion.** `features/engine.py`: EWMA at two spans, short-window slope, count of processes above a CPU threshold, and — **only where `native`** per Phase 1 — the `load_avg_1 / cpu_percent` ratio. Add a test asserting every feature at time *t* is computable from samples ≤ *t*.

### Success Criteria

**Automated (both platforms):**
- [ ] Feature column lists are identical between training and live paths for a given model version
- [ ] `predict()` reconstructs absolute units correctly from a delta model
- [ ] Every feature is causal: no future sample changes a past feature value
- [ ] `selected == "lightgbm"` for `cpu_percent`/short **and** `mem_percent`/short — xfail flips to pass
- [ ] Holdout directional accuracy > 0.55 for both
- [ ] Holdout MAE beats naive persistence by ≥ the configured margin
- [ ] Windows: the same criteria hold on the Windows metric set, **without** `load_avg`

**Manual:**
- [ ] Training log shows LightGBM winning with the margin printed
- [ ] **Full E2E run**

**Pause for confirmation.**

---

## Phase 5 — Stress harness (spawn-safe)

### Overview
Produces genuine contention on real hardware in minutes. Addresses the root cause behind F1 and F3, and W4 by construction.

### Changes Required

**1. The harness.** New `src/aiperfopt/stress/` package. Safety contract, non-negotiable:
- **CPU and disk only.** No memory allocation, no network flooding.
- CPU: at most `cpu_count − 1` busy child processes, ramped.
- Disk: bounded temp-file write/read under `AIPERFOPT_HOME/stress-scratch`, hard byte ceiling, always cleaned up.
- Global max duration (default 180s), hard wall-clock kill, cleanup on interrupt and `atexit`. Ctrl-C must fully release the machine.
- Refuses to start if another session is active.

**Windows-safe by design (W4):** worker targets are importable top-level functions with picklable arguments under a `__main__` guard, so `spawn` works; teardown goes through `psutil.Process.terminate()` then `.kill()` with a timeout, never raw signals or `SIGKILL`.

Scenarios: `cpu_ramp`, `cpu_spike`, `disk_burst`, `cpu_plus_disk`, `idle_baseline`.

**2. Session manifests as ground truth.** New `stress_sessions` table (`started_at`, `ended_at`, `scenario`, `parameters`, `platform`) across `storage/schema.py`, `records.py`, `writer.py`, `reader.py`. Label construction uses real windows instead of inferring from thresholds.

**3. Train on real data.** `retraining/real_usage_data.py`: prefer stress-session windows for positive labels; calibrate normality from `idle_baseline` plus passive collection.

**4. Mark the unvalidated categories.** Under the CPU+disk-only decision, `MEMORY_PRESSURE_GRADUAL` and `MEMORY_PRESSURE_SUDDEN` cannot be validated on real data. Add `validated_on_real_data: bool` per category in `diagnosis/taxonomy.py`, surfaced in Phase 8's UI and in `reporting/summary.py`. They are kept and labelled, not presented as equal to the CPU/disk categories.

### Success Criteria

**Automated (both platforms):**
- [ ] Harness cleans up every child process and temp file on normal exit
- [ ] Harness cleans up on interrupt mid-run
- [ ] No memory scenario is registered (the constraint is structural, not documentary)
- [ ] A session writes a manifest with both timestamps and the platform
- [ ] Real-usage labeling picks up positives inside a manifest window
- [ ] Windows: workers start under `spawn` and terminate without signals

**Manual:**
- [ ] `aiperfopt stress cpu_ramp` visibly raises CPU; the machine returns to normal within seconds of exit
- [ ] Ctrl-C releases the machine immediately
- [ ] After `cpu_ramp` + `idle_baseline`, real training data contains both classes
- [ ] **Full E2E run**

**Pause for confirmation.**

---

## Phase 6 — Quality gate and fitted thresholds

### Overview
Fixes F3's deeper problem — nothing can currently fail — and replaces three hyperparameters chosen by argument with values fitted from data.

### Changes Required

**1. Define skill.** `retraining/comparison.py`: a `SkillGate` evaluated per (metric, horizon) on a holdout the candidate never trained on.
- Forecasting: `MAE < persistence_MAE × (1 − margin)` **and** `directional_accuracy > 0.5 + margin`.
- Contention: F1 must beat a **threshold-rule baseline** ("predict contention if current value > labeling threshold"). This baseline does not exist today and must be added — with no comparator, `f1 0.0` looked acceptable.

The gate is on *the deployed forecaster*, whichever it is — not on "LightGBM must win". A baseline that genuinely beats persistence is legitimately deployable; the comparison payload separately records whether LightGBM won, so Phase 4's goal stays measurable without ever locking the system out entirely.

**2. Make it block.** `retraining/deployment.py`: `confirm_deployment` raises `ModelDeploymentError` when the gate fails, **even with `confirmed=True`**. Human confirmation stays necessary and stops being sufficient. The router maps this to 422 enumerating the failed criteria.

**3. Fit the thresholds.**
- `contention.decision_threshold` (today `0.75`, chosen in a config comment): select from a precision-recall curve on holdout against a stated precision floor, honouring the original precision-over-recall intent. Persist the curve alongside the model so the choice is auditable.
- `contention.anomaly_significant_threshold` (today `0.7`): set as a high quantile (e.g. 99th percentile) of the score distribution on **real normal** data, per platform. A fixed constant cannot be right for both a Mac and a Windows box.
- `contention.anomaly_dampening_factor` (today `0.3`): **make it earn its place.** Add a test measuring holdout precision with dampening on versus off. If it does not improve precision, delete it. Today it only ever subtracts 30% from every prediction.

**4. Make it fail CI.** New `tests/test_skill_gate.py`: fixed-seed training asserting the gate passes for `cpu_percent`/short and `mem_percent`/short. This failing is the signal that a change destroyed model quality — impossible to detect today.

**5. Report it.** Gate results in `RetrainingComparisonOut` (`api/schemas.py`); `frontend/widgets/retraining_panel.py` shows pass/fail per metric with numbers.

### Success Criteria

**Automated (both platforms):**
- [ ] A deliberately-crippled candidate (shuffled labels) fails the gate
- [ ] `confirm_deployment(confirmed=True)` on a failing candidate **raises**
- [ ] `POST /retraining/confirm` on a failing candidate returns 422 listing failed criteria
- [ ] The contention threshold-rule baseline is computed and persisted
- [ ] The fitted decision threshold is recorded with its PR curve
- [ ] The anomaly threshold is a fitted quantile, and differs per platform where the distributions differ
- [ ] A test decides the dampening factor's fate on measured precision
- [ ] `tests/test_skill_gate.py` passes on a fixed seed, on both platforms

**Manual:**
- [ ] Retraining panel shows per-metric gate pass/fail with numbers
- [ ] Deploying a bad model is refused with a readable reason
- [ ] **Full E2E run**

**Pause for confirmation.**

---

## Phase 7 — Persist forecasts

### Overview
Fixes F5. Small change, disproportionate payoff: it is what makes the central claim demonstrable.

### Changes Required
`api/live_state.py`: in `_tick`, after building the contention input row, write one `ForecastPredictionRecord` per deployed metric — `generated_at`, `target_timestamp = generated_at + horizon_seconds`, `predicted_value`, `confidence`, `model_version`. `insert_forecast_predictions_batch` already exists (`storage/writer.py:155`); storage needs nothing new. Add the forecast values to `LiveState` and `LiveStateOut` so the WebSocket payload carries them for Phase 8's chart.

### Success Criteria

**Automated (both platforms):**
- [ ] One live tick with N deployed forecasters writes N forecast rows — the Phase 2 xfail flips to pass
- [ ] `target_timestamp == generated_at + horizon_seconds`
- [ ] After ticks plus later real samples, `/reporting/predicted-vs-actual` returns non-empty paired points
- [ ] `/reporting/summary` `prediction_accuracy` is populated

**Manual:**
- [ ] After ~5 minutes live, the Predicted vs. Actual tab shows two diverging-then-converging lines
- [ ] **Full E2E run**

**Pause for confirmation.**

---

## Phase 8 — UI: charts, the segfault, and platform-correct text

### Overview
Fixes F6, F9 and the presentation half of W2.

### Changes Required

**1. Fix the crash.**
- `frontend/live_client.py:55`: parent the socket — `QWebSocket(parent=self)`. Today the `QTimer` is parented and the socket is not, leaving C++ destruction order undefined while signals may be in flight.
- `tests/frontend/server_fixture.py:76`: stop the live loop and monitoring pipeline explicitly *before* signalling uvicorn, and **raise** if the thread fails to join rather than continuing into the next test with live threads running.
- `api/app.py:87`: guard `asyncio.run_coroutine_threadsafe` against a closed loop and drop the broadcast if the loop is gone.

**2. A live trend chart with the forecast on it.** New `frontend/widgets/live_chart.py`, modelled on the existing QtCharts usage at `frontend/widgets/predicted_vs_actual.py:14` — same imports, same axis pattern. A rolling window of actual CPU and memory as solid lines, with the current forecast as a **dashed continuation past "now"** plus a confidence band from the model's confidence scalar. This is the wow moment: the solid line goes where the dashed line said it would.

**3. Put it on the dashboard.** `frontend/widgets/dashboard.py`: insert between the tile row and the diagnosis frame. Keep the tiles.

**4. Honest empty states.** Consume Phase 0's `/readiness`. Replace the permanently-green "Live" banner (`dashboard.py:132`) with the real state: `Collecting data — 2m 15s of 5m needed`, `No model trained yet — run "aiperfopt train"`, `Model deployed but below skill gate`, or `Live`. Mark memory-category diagnoses with the Phase 5 unvalidated flag. Show the Phase 1 capability note where a metric is emulated on this platform.

### Success Criteria

**Automated (macOS):**
- [ ] `QT_QPA_PLATFORM=offscreen pytest tests/frontend -q` exits **0**, not 139, three consecutive runs — the Phase 2 xfail flips to pass
- [ ] Full `pytest -q` completes with no fatal error
- [ ] New test: the live chart renders actual and forecast series from a `LiveStateOut` payload
- [ ] New test: each readiness state renders its own banner text
- [ ] New test: `server_fixture` raises if the server thread fails to join

**Automated (Windows — CI):**
- [ ] Frontend suite exits 0 headless
- [ ] Capability note renders for `load_avg` as emulated

**Manual:**
- [ ] `aiperfopt run`, then `aiperfopt stress cpu_ramp` in another terminal: the forecast line rises **before** the actual line
- [ ] A fresh install shows "no model trained yet", not a green "Live"
- [ ] **Full E2E run on both platforms**

**Pause for confirmation.**

---

## Phase 9 — Storage and overhead

### Changes Required
- `monitoring/collectors.py`: store the top-N processes per sample by resource use (default 20) plus one aggregate row for the remainder, instead of all ~180. Process data exists to attribute contention; the tail contributes nothing. Fixes F10.
- `config/default.yaml`, `storage/rollup.py`: separate, shorter retention for process data (default 6h) from system-metric retention (72h). Review indexes against the actual reporting queries.
- New `tests/test_overhead.py`: measure this process's own CPU over a fixed monitoring window with the live loop running, and assert against a documented budget. Asserted nowhere today.

### Success Criteria

**Automated (both platforms):**
- [ ] Only top-N processes plus one aggregate row are stored per sample
- [ ] Process retention prunes on its own shorter schedule
- [ ] Measured self-CPU stays under the documented budget, on each platform separately

**Manual:**
- [ ] 1h run: DB growth is an order of magnitude below the current trajectory
- [ ] Contributing-process attribution still names the right process during `cpu_ramp`
- [ ] **Full E2E run**

**Pause for confirmation.**

---

## Phase 10 — Revive the dead surfaces honestly

### Overview
Verifies Phase 3 actually paid off. Every category and the recommendation engine must be provably reachable.

### Changes Required
- New `tests/test_all_categories_reachable.py`: drive each of the 7 actionable categories end to end through to a persisted recommendation. Today all 7 are unreachable and no test notices.
- `config/recommendations.yaml`: verify every actionable category has entries **for both platforms**, and that the loader errors loudly on a missing one — the file's comments claim this; assert it.
- `reporting/summary.py`: verify accuracy, diagnosis counts and deployment history populate from a real run; mark memory categories unvalidated; record the platform in the report.
- `README.md`: rewrite around the CLI, with separate Windows and macOS quickstarts. The current "Running the full pipeline" snippet is a multi-line Python blob no user should paste.

### Success Criteria

**Automated (both platforms):**
- [ ] All 7 actionable categories reach a persisted recommendation
- [ ] A category missing a platform variant raises at load
- [ ] `/reporting/summary` returns populated accuracy and non-zero diagnosis counts after a real run
- [ ] Full `pytest -q` green, no fatal errors, on Windows and macOS

**Manual:**
- [ ] `cpu_ramp` produces a CPU diagnosis with a sensible recommendation naming the real culprit process — on both platforms
- [ ] Weekly report shows real accuracy numbers
- [ ] A new user on Windows can follow the README start to finish without writing Python
- [ ] **Full E2E run on both platforms**

---

## Testing Strategy

### The gap this closes
355 tests pass while every model has zero skill (F11). The suite tests plumbing — functions called, records written, threads started — and asserts nothing about quality. Phase 2 labels this honestly; every later phase adds at least one test that fails when *behaviour* degrades, not merely when code errors.

### New test categories
1. **Train/serve equivalence** — identical input through the training and live paths must yield identical feature vectors and identical scores. Catches the whole F1/F4 class.
2. **Skill assertions** — fixed-seed training must clear the gate. Catches silent model regression.
3. **Distribution assertions** — the detector must not flag its own training data, nor ordinary real data. Catches F1 directly.
4. **Reachability** — every category and the recommendation engine must be provably reachable. Catches F2.
5. **Causality** — no feature may depend on a future sample. Guards the leakage the code worries about only in prose.
6. **Lifecycle** — no test may leak a live thread past teardown. Catches F6.
7. **Platform parity** — the same behavioural suite runs on Windows and macOS, and any metric the platform only emulates is excluded from models rather than silently substituted. Catches W1.

### Manual verification loop
Repeatable end-to-end check, run on **both platforms** after every phase from Phase 5 onward:
1. `aiperfopt doctor` → healthy, capability table correct for this OS
2. `aiperfopt collect --minutes 15` → real normal data
3. `aiperfopt stress cpu_ramp` and `aiperfopt stress idle_baseline`
4. `aiperfopt train --source real --deploy` → gate passes, LightGBM selected
5. `aiperfopt run`
6. `aiperfopt stress cpu_ramp` in a second terminal → forecast line rises before actual, diagnosis names the real process, recommendation appears with platform-correct wording
7. Predicted vs. Actual tab shows the prediction leading the event

## References

- Anomaly imputation skew: `src/aiperfopt/anomaly/detector.py:83`, `:100`, `:111`
- Diagnosis short-circuit: `src/aiperfopt/diagnosis/taxonomy.py:228`
- Anomaly dampening: `src/aiperfopt/contention/prediction.py:59`
- Excluded `current_value`, two copies: `src/aiperfopt/forecasting/training_pipeline.py:265`, `src/aiperfopt/forecasting/evaluation.py:32`
- Counter-metric precedent for the same bug class: `src/aiperfopt/forecasting/dataset.py` docstring
- Forecasts never persisted: `src/aiperfopt/api/live_state.py:163`
- Emulated load average enters the model: `src/aiperfopt/monitoring/collectors.py:84`, `src/aiperfopt/features/config.py:54`
- macOS-only user advice: `config/recommendations.yaml:47`, `:77`, `:143`; `src/aiperfopt/diagnosis/text.py:112`, `:118`
- Mixed priority scales: `src/aiperfopt/monitoring/process_tracker.py:128`
- Unparented socket: `src/aiperfopt/frontend/live_client.py:55`
- Leaked test threads: `tests/frontend/server_fixture.py:76`
- Unguarded broadcast: `src/aiperfopt/api/app.py:87`
- Chart precedent to model on: `src/aiperfopt/frontend/widgets/predicted_vs_actual.py:14`
