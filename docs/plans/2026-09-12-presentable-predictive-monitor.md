# Presentable Predictive Monitor — Review 2 Plan

**Date:** 2026-09-12
**Status:** proposed
**Supersedes:** `2026-08-24-make-it-actually-predict.md` (Phase 0 of that plan is complete and retained; its stress-harness strategy is deleted)

---

## Overview

The previous plan assumed the models had no predictive skill because the machine was too idle to produce anything worth learning, and proposed a **stress harness** that would deliberately pin CPU and hammer disk to manufacture training data. That diagnosis was wrong, and the remedy would have stressed the development machine for no reason.

The real cause is narrower and far more fixable: **the system already collects the most predictive data it has, and discards it before the model ever sees it.**

This plan pivots from *manufacturing load* to *stopping the discard*. It keeps the project identity, the architecture, and all eleven modules intact, and ends with a presentable product that answers every Review 2 requirement with measured evidence.

---

## The Core Finding

Per-process data is collected every 3 seconds and stored: pid, name, CPU, memory RSS/VMS, thread count, priority, status, and `create_time` (`storage/records.py`). A complete, tested **process launch/termination detector** already exists from Phase 2 (`ProcessEvent`, `monitoring/process_tracker.py`).

The feature engineering layer uses **none of it**:

```python
# features/config.py:54
gauge_metrics: tuple[str, ...] = ("cpu_percent", "mem_percent", "swap_percent", "load_avg_1")
counter_metrics: tuple[str, ...] = ("disk_read_bytes", "disk_write_bytes",
                                    "net_bytes_sent", "net_bytes_recv")
```

System-level aggregates only. `grep -rn "process" src/aiperfopt/features/` returns nothing.

So the model is asked to predict *future CPU* from *past CPU* and nothing else — exactly the information the naive persistence baseline already has. That is a complete explanation for why LightGBM loses to persistence on 4 of 4 metric-horizons, and it requires no theory about the machine being idle.

**A process launching is a leading indicator.** Chrome opening, a build starting, a video call joining — system CPU rises *after* the launch, and the launch is observable *before* the rise. Normal daily work generates these events constantly. That signal is already in the database, timestamped, unused.

### Why this is the right pivot

| | Old approach | This plan |
|---|---|---|
| Source of signal | Manufacture load artificially | Use signal already being collected and thrown away |
| Cost to the machine | Deliberate CPU + disk saturation | None — passive observation of normal use |
| Honesty of result | Model learns a synthetic stressor | Model learns real workload behaviour |
| Review narrative | "We built 11 modules" | "We diagnosed why it had no skill, found it was blind to its most predictive input, fixed it, and measured the gain" |

---

## Current State

Phase 0 of the superseded plan is complete and holds: packaged config, single application home, `aiperfopt` CLI, `/readiness`, pinned Python 3.12, 386 backend tests passing locally.

Outstanding problems this plan addresses:

| Ref | Problem | Consequence |
|---|---|---|
| **F1** | Anomaly detector fills NaN with column means in `fit` (`detector.py:83`) but `0.0` in `score`/`score_batch` (`:100`, `:111`) | Every real sample scores `1.0` |
| **F2** | Diagnosis tests anomaly first (`taxonomy.py:228`), so a `1.0` score always returns `UNUSUAL_PATTERN_ANOMALY` | 7 actionable categories, the SHAP explainer and the entire recommendation engine are dead at runtime |
| **F3** | No quality gate — holdout `f1 0.0` deployed with `regressed: False` | Nothing can fail; a broken model ships silently |
| **F4** | Model never sees `current_value`; no process features at all | LightGBM loses to persistence 4/4 |
| **F5** | `live_state._tick` computes forecasts and never persists them | `forecast_predictions` has 0 rows; Predicted-vs-Actual and report accuracy permanently empty |
| **F6** | Qt suite segfaults (exit 139) — unparented `QWebSocket` (`live_client.py:55`), leaked threads (`server_fixture.py:76`) | Cannot show a clean test run |
| **F9** | Dashboard is three number tiles; no trend, no forecast | Nothing visually demonstrates "predictive" |
| **F10** | Every process stored on every poll — 26,451 rows / 5 min ≈ **7.6M/day** | The collector itself would stress the machine over days of collection |
| **Pending** | The `_cmd_doctor` exception-guard fix is uncommitted; Windows CI red and undiagnosable | CI is not trustworthy |

---

## Decisions Taken

| Question | Decision |
|---|---|
| Timeline | No fixed deadline — sequence by value, presentable at every phase boundary |
| Platform | **macOS only.** Code stays portable by construction; Windows CI leg removed |
| Training data | **Passive collection over several days of normal use.** No artificial load, ever |
| Live demo | **Replay of a recorded real session** — deterministic, repeatable, zero load while presenting |
| Headline model | **Contention prediction (classification)** — "will pressure occur in the next 60s" |
| Machine impact | Explicit, tested overhead budget. The monitor must not be a burden |

---

## Desired End State

- `aiperfopt` runs passively for days at **under 2% average CPU** of its own, with a bounded database.
- The contention classifier **beats a threshold-rule baseline** on a chronological holdout it never trained on, with process-derived leading indicators as the measured reason.
- An **ablation** isolates that gain: identical data, identical split, identical model — system-only features vs system+process features.
- The anomaly detector flags ordinary usage **rarely**, so all 7 actionable diagnosis categories and the recommendation engine are reachable.
- A **hard quality gate** refuses to deploy a model that fails to beat its baseline.
- The dashboard shows a **live trend with the forecast projected ahead**, and Predicted-vs-Actual shows the prediction leading the event.
- `aiperfopt evaluate --report` regenerates **every number and figure** used in the presentation.
- `aiperfopt replay` reproduces a real pressure event on demand, without loading the machine.

## What We're NOT Doing

- **No stress harness. No artificial CPU, disk, memory or network load of any kind.** This is deleted from scope, not deferred.
- No Windows verification, and no Windows CI job. The code remains portable; we simply do not claim or test it.
- No memory-pressure induction or swap thrashing.
- No frontend rewrite. PySide6 and the existing four tabs stay.
- No deep learning, no second model family. LightGBM gets the inputs it was missing; that is the experiment.
- No new architecture and no new modules beyond replay and process-feature extraction. Same eleven subsystems.
- No attempt to restore the original README's claimed test count.
- No per-prediction quantile calibration; confidence stays a validation-derived scalar.

---

## Phase Order

Dependency-driven, with one scheduling constraint that dominates everything.

> **Collection is on the critical path and costs wall-clock time, not work.**
> Phase 2 makes collection lightweight and then **starts it running in the background for the remainder of the project**. Data accumulates while Phases 3–9 are built. Delaying Phase 2 delays every result.

| # | Phase | Why here |
|---|---|---|
| 1 | Stabilise the baseline | CI must be trustworthy before it can verify anything |
| 2 | **Lightweight collection → start collecting** | Critical path. Days of wall-clock; everything downstream needs this data |
| 3 | Unblock the anomaly detector | Highest leverage — revives 8 dead surfaces |
| 4 | **Process-level leading indicators** | The pivot itself |
| 5 | **Train, evaluate, ablate** | The Results and Analysis centrepiece |
| 6 | Quality gate | Gives the results integrity |
| 7 | Persist forecasts | Unlocks Predicted-vs-Actual and report accuracy |
| 8 | Replay mode | The live demonstration mechanism |
| 9 | UI charts and segfault fix | Makes it visually presentable |
| 10 | Evidence pack | Assembles the six Review 2 answers |

---

## Phase 1 — Stabilise the baseline

### Changes Required
1. Commit the pending `_cmd_doctor` exception-guard fix (currently uncommitted in `src/aiperfopt/cli.py`).
2. Remove the `windows-latest` legs from `.github/workflows/ci.yml`; keep `macos-latest` backend, frontend, and the wheel-packaging job.
3. Diagnose and fix the **macOS** backend CI failure. It fails on the `doctor` step with exit code 1, which is the signature of an uncaught exception — `_cmd_doctor` itself only ever returns 0 or 2. The committed guard should surface the real traceback in the job output.
4. Remove `continue-on-error` from the frontend job only once Phase 9 fixes the segfault; until then, annotate it in the workflow as a known-red job rather than a silent pass.

### Success Criteria
**Automated:**
- [ ] `pytest --ignore=tests/frontend -q` → 386+ passed locally
- [ ] macOS backend CI job is **green**
- [ ] Wheel-packaging CI job green
- [ ] No Windows job present in the workflow

**Manual:**
- [ ] CI status on the branch is meaningful — every non-green job has a known, written reason

---

## Phase 2 — Lightweight collection, then start collecting

### Overview
Makes multi-day passive collection safe for the machine, then begins it. **Start the collector at the end of this phase and leave it running.**

### Changes Required

**1. Bounded process storage.** `monitoring/process_tracker.py`: store the **top 20 processes by CPU plus top 20 by memory RSS (unioned), plus one aggregate row** summarising everything else (`other_count`, `other_cpu_sum`, `other_mem_sum`). Process data exists to attribute contention and to derive leading indicators; the long tail of idle processes contributes neither. Target: ~21 rows per poll instead of ~180 — an order-of-magnitude reduction.

Launch/terminate **events** are unaffected and continue to be recorded for **every** process. Events are small, rare, and are precisely the leading indicator Phase 4 depends on — truncating them would defeat the pivot.

**2. Separate process retention.** `resources/default.yaml`, `storage/rollup.py`: process rows get their own, shorter retention (default **6 hours**) independent of system-metric retention (72 hours). Process detail is needed for recent attribution and for training-window extraction, not indefinitely.

**3. Training-set export.** Because process detail is pruned at 6 hours but training needs days, add `aiperfopt export-training-window --since <ts> --out <file>` which writes the derived **process summary series** (Phase 4's aggregates, not raw per-process rows) to durable storage on a rolling basis. Wire a periodic call into the ingestion consumer's existing retention trigger so summaries are computed *before* the raw rows are pruned.

**4. Overhead budget, measured.** New `tests/test_overhead.py`: run the monitoring pipeline for a fixed window and assert this process's own mean CPU stays under a documented budget (**2%**), and that database growth per hour stays under a documented ceiling. Currently asserted nowhere.

**5. Begin collection.** `aiperfopt collect` runs in the background from here to the end of the project, across normal daily work.

### Success Criteria
**Automated:**
- [ ] At most 41 process rows + 1 aggregate row are written per poll
- [ ] The aggregate row's counts reconcile with the untruncated enumeration
- [ ] Process launch/terminate events are recorded for processes *outside* the top-N
- [ ] Process rows prune on their own 6-hour schedule, independent of system metrics
- [ ] Process summary rows survive raw-row pruning
- [ ] Measured self-CPU is under the 2% budget

**Manual:**
- [ ] One hour of collection produces a database an order of magnitude smaller than the current trajectory
- [ ] Machine is subjectively unaffected — no fan, no lag
- [ ] **Collector is running and will stay running**

---

## Phase 3 — Unblock the anomaly detector

### Overview
Fixes F1 and F2. Highest single leverage in the plan: revives the recommendation engine, the SHAP explainer, and all 7 actionable diagnosis categories.

### Changes Required

**1. One imputation, used everywhere.** `anomaly/detector.py`: persist training column means on the artifact; add a single private `_prepare(X)` that reindexes to `feature_columns` and fills from those means, falling back to `0.0` only for columns entirely NaN in training. Call it from `fit`, `score`, and `score_batch`. Delete the two divergent `fillna(0.0)` sites. Bump the artifact schema version; `load_anomaly_detector` must **reject** an artifact lacking stored means rather than loading one that behaves like today.

**2. Calibrate against real normality.** The detector must be fit on data from the machine it scores. Add a calibration path over **real collected feature vectors** from Phase 2. Synthetic data may contribute, but real normal data must be present or the detector refuses to fit for live use.

**3. Stop the anomaly branch short-circuiting diagnosis.** `diagnosis/taxonomy.py`: evaluate contention triggering and resource dominance **first**. `UNUSUAL_PATTERN_ANOMALY` becomes the fallback when nothing else resolves, and a high anomaly score becomes a **flag on the record** rather than a category that pre-empts all others.

### Success Criteria
**Automated:**
- [x] `score(row)` and `score_batch(row.to_frame().T)` return identical values (`tests/test_anomaly_detector.py::test_score_batch_matches_individual_scores`)
- [x] Scoring the training matrix yields mean score < 0.3 (`test_scoring_training_matrix_has_low_mean_score`, and again on real data in `tests/test_anomaly_calibration.py`)
- [x] On held-out **real** normal data, fewer than 10% of rows exceed the significance threshold (`tests/test_anomaly_calibration.py::test_calibration_fits_on_real_data_and_generalizes_to_holdout`)
- [x] High anomaly + triggered CPU contention resolves to a **CPU** category, not the anomaly category (`tests/test_diagnosis_service.py::test_high_anomaly_with_triggered_cpu_contention_resolves_to_cpu_category`, `tests/test_diagnosis_taxonomy.py::test_triggered_resource_category_wins_over_high_anomaly`)
- [x] Each of the 7 actionable categories is reachable from a constructed contention record (`tests/test_diagnosis_taxonomy.py`, unaffected by anomaly score after the reorder)
- [x] Loading a pre-Phase-3 artifact raises (`tests/test_anomaly_persistence_and_notifications.py::test_load_pre_phase3_artifact_raises`)

**Manual:**
- [ ] During ordinary use the "unusual behaviour" banner is **absent** — not yet observable: no model is deployed on this machine (`aiperfopt doctor` reports `needs_training`), so there is no live diagnosis surface to watch yet. Revisit once Phase 5/6 deploy a real model against the now-calibratable detector.
- [ ] Recommendations table receives rows for the first time — same blocker as above.

Full backend suite: 419 passed (up from 409 at the end of Phase 2), verified locally (`pytest --ignore=tests/frontend -q`).

---

## Phase 4 — Process-level leading indicators (the pivot)

### Overview
Gives the model the information it has been blind to. This is the technical heart of the project and the basis of the Results slide.

### Design

Rather than bolting a parallel pipeline alongside the existing one, extract a **process summary series** aligned to the system-metric timeline, then feed those columns through the **existing** `compute_feature_matrix` as additional gauge metrics. This preserves the codebase's genuinely good property — one vectorised function shared by training and live inference, structurally preventing train/serve skew — and it means lags, rolling statistics and rate-of-change come for free on every new feature.

**New module:** `src/aiperfopt/features/process_features.py`

Per system-metric timestamp, derived from `raw_process_metrics` and `process_events`:

*Event-derived — the true leading indicators:*
- `proc_launch_count_30s`, `proc_launch_count_120s`
- `proc_terminate_count_30s`
- `proc_launch_burst_10s` — max launches in any 10s sub-window (detects application start-up storms)

*Young-process features — the strongest single signal:*
- `proc_young_count` — processes with `create_time` inside the last 60s
- `proc_young_cpu_sum` — CPU attributable to processes launched in the last 60s

  *A newly launched process consuming CPU now is the clearest available predictor of sustained CPU 30–60s ahead, and it is information persistence cannot possess.*

*Concentration and breadth — distinguishes one runaway from broad load:*
- `proc_cpu_sum`, `proc_cpu_top1`, `proc_cpu_top1_share`, `proc_cpu_top5_share`
- `proc_count_above_20pct`
- `proc_count_active`

*Memory and threads:*
- `proc_mem_rss_sum`, `proc_mem_top1_rss`, `proc_threads_sum`

**Wiring:** `FeatureEngineeringConfig.gauge_metrics` is extended with the process summary column names, so the existing lag/rolling/rate machinery applies unchanged. `FEATURE_SET_VERSION` bumps to `v2`; a model artifact records its feature-set version and refuses to load against a mismatched one.

**Live path:** `features/storage_bridge.py` and `contention/dataset.py` must build the process summary for the live window exactly as training does. The SHAP explainer (`diagnosis/explainer.py`) builds its own rows and must carry the same columns, or attribution explains a different vector than the model scored.

### Success Criteria
**Automated:**
- [ ] Process summary columns are **identical** between the training path and the live path for the same window — asserted directly, not by inspection
- [ ] **Causality:** every feature at time *t* is computable from samples with timestamp ≤ *t*; no future sample alters a past feature value
- [ ] `proc_young_cpu_sum` is non-zero in a window containing a launch with CPU, and zero in a window without
- [ ] A gap in process sampling yields `None`, never a stale carried-forward value
- [ ] Loading a `v1` feature-set model against a `v2` matrix raises
- [ ] SHAP explainer rows carry the full `v2` column set

**Manual:**
- [ ] Launching an application visibly moves the young-process features in a live feature vector

---

## Phase 5 — Train, evaluate, ablate

### Overview
Turns the pivot into a measured result. This phase produces the Review 2 "Results and Analysis" content.

### Changes Required

**1. Reframe the headline target.** Contention prediction (classification: "will resource pressure occur within the short horizon") becomes the primary deliverable, with forecasting retained as secondary. Rationale: it is what the product is *for*, it is what the leading indicators actually help, and precision/recall/PR-AUC are far more meaningful to present than an MAE on a smooth series where persistence is near-optimal by construction.

**2. Handle class imbalance honestly.** Real passive data will be heavily negative. Report the **base rate** explicitly, use class weighting, select the decision threshold from a **precision-recall curve** on holdout rather than a config constant, and evaluate against **PR-AUC** rather than accuracy. An accuracy figure on an imbalanced binary problem is meaningless and will not be presented.

**3. Give the forecaster its missing input too.** Remove `current_value` from the excluded-column set (`training_pipeline.py:265` and its duplicate `_NON_FEATURE_COLUMNS` in `evaluation.py:32` — collapse to one exported constant). Predict `target − current_value` (a delta) and reconstruct absolute values in `predict`, since trees cannot extrapolate an absolute level. Keep all evaluation in absolute units so numbers stay comparable to the baselines.

**4. Add the baseline that was missing.** Contention currently has nothing to be compared against, which is why `f1 0.0` looked acceptable. Add a **threshold-rule baseline**: "predict contention if the current value already exceeds the labeling threshold."

**5. The ablation — the centrepiece.** New `aiperfopt evaluate --ablation`:

| Arm | Features | Purpose |
|---|---|---|
| A | System-only (`v1`) | Reproduces the current, skill-less model |
| B | System + process (`v2`) | The pivot |

Identical data, identical chronological split, identical model and hyperparameters. The delta between A and B **is** the measured contribution of this project's central idea. Report PR-AUC, F1 at the fitted threshold, precision, recall, and forecast MAE vs persistence for both arms.

**6. Reproducible artifacts.** `aiperfopt evaluate --report` writes to `results/`:
- `dataset_summary.txt` — hours collected, sample count, positive base rate, date range
- `ablation.csv` — the A/B table
- `baseline_comparison.csv` — model vs every baseline, per metric/horizon
- `pr_curve_contention.png`, `confusion_matrix.png`
- `feature_importance.png` — SHAP, expected to show process features ranking highly
- `overhead.txt` — measured monitor footprint from Phase 2

### Success Criteria
**Automated:**
- [ ] Holdout is strictly chronological — no future-to-past leakage, asserted
- [ ] The threshold-rule baseline is computed and persisted
- [ ] The decision threshold is fitted from the PR curve and stored with the curve
- [ ] `predict()` reconstructs absolute units correctly from a delta model
- [ ] `--ablation` produces both arms from one command on one dataset
- [ ] `--report` regenerates every artifact from scratch

**Manual:**
- [ ] **Arm B beats Arm A on PR-AUC**, and the margin is stated plainly
- [ ] Process features appear among the top SHAP contributors
- [ ] If Arm B does *not* beat Arm A, that is reported honestly as the finding, with the analysis of why — a negative result rigorously obtained is presentable; a fabricated positive is not

---

## Phase 6 — Quality gate

### Overview
Gives the Phase 5 numbers integrity. Currently nothing can fail.

### Changes Required
1. `retraining/comparison.py`: a `SkillGate` evaluated per target on the chronological holdout.
   - **Contention:** must beat the threshold-rule baseline's F1 by a configured margin.
   - **Forecasting:** MAE below persistence by a margin **and** directional accuracy above `0.5 + margin`.
   The gate is on the *deployed* model, whichever it is — a baseline that genuinely wins is legitimately deployable. The comparison payload separately records whether LightGBM won, so Phase 5's goal stays measurable without ever locking deployment out entirely.
2. `retraining/deployment.py`: `confirm_deployment` raises `ModelDeploymentError` when the gate fails **even with `confirmed=True`**. Human confirmation remains necessary and stops being sufficient. The router maps this to 422 listing the failed criteria.
3. New `tests/test_skill_gate.py`: fixed-seed training asserting the gate passes. Its failure is the signal that a change destroyed model quality — undetectable today.
4. Surface gate results in `RetrainingComparisonOut` and the retraining panel.

### Success Criteria
**Automated:**
- [ ] A deliberately crippled candidate (shuffled labels) **fails** the gate
- [ ] `confirm_deployment(confirmed=True)` on a failing candidate raises
- [ ] `POST /retraining/confirm` returns 422 enumerating failed criteria
- [ ] `tests/test_skill_gate.py` passes on a fixed seed

**Manual:**
- [ ] Retraining panel shows per-target pass/fail with the actual numbers
- [ ] Attempting to deploy a bad model is refused with a readable reason

---

## Phase 7 — Persist forecasts

### Overview
Fixes F5. Small change, disproportionate payoff — it is what makes the central claim visible.

### Changes Required
`api/live_state.py`: in `_tick`, after building the contention input row, write one `ForecastPredictionRecord` per deployed metric — `generated_at`, `target_timestamp = generated_at + horizon_seconds`, `predicted_value`, `confidence`, `model_version`. `insert_forecast_predictions_batch` already exists (`storage/writer.py:155`); storage needs no change. Add forecast values to `LiveState` and `LiveStateOut` so the WebSocket payload carries them for Phase 9's chart.

### Success Criteria
**Automated:**
- [ ] One live tick with N deployed forecasters writes N forecast rows
- [ ] `target_timestamp == generated_at + horizon_seconds`
- [ ] After ticks plus later real samples, `/reporting/predicted-vs-actual` returns non-empty paired points
- [ ] `/reporting/summary` `prediction_accuracy` is populated

**Manual:**
- [ ] After a few minutes live, the Predicted vs. Actual tab shows two lines

---

## Phase 8 — Replay mode (the demonstration mechanism)

### Overview
Makes the live demonstration deterministic, repeatable, and completely free of machine load. A recorded real pressure event can be reproduced on demand instead of hoping one occurs while presenting.

### Design
Replay **substitutes for the collector**: the monitoring engine is not started, and recorded samples are injected into storage with timestamps rewritten so the session appears to be happening now.

**New module:** `src/aiperfopt/replay/`

- `aiperfopt record --minutes N --out session.json` — captures a window of system samples, process samples and process events into a portable file. (In practice sessions are selected retrospectively from the Phase 2 collection by time window; `record` is the explicit path.)
- `aiperfopt replay --session session.json [--speed N]` — replays into a **separate replay database**, never the primary one, with timestamps shifted so the first sample maps to "now minus session length" and subsequent samples advance at `speed` × real time. The live inference loop then runs over it unmodified.

Constraints:
- Replay must refuse to run if the monitoring pipeline is active, to prevent real and replayed samples interleaving.
- The UI must display an unmistakable **"REPLAY"** indicator whenever it is serving replayed data. The demo is honest about what it is — a recording of real behaviour, not a simulation and not live capture.
- Speed multipliers must not break feature causality: lag windows are computed on the rewritten timestamps, so a 10× replay still yields the same feature values as real time.

### Success Criteria
**Automated:**
- [ ] A recorded session replays to produce **identical feature vectors** to the original, at 1× and at 10×
- [ ] Replay writes nothing to the primary database
- [ ] Replay refuses to start while monitoring is running
- [ ] A replayed contention event produces the same prediction and diagnosis as the original recording

**Manual:**
- [ ] A recorded pressure event replays end-to-end with the REPLAY badge visible
- [ ] Machine load during replay is negligible

---

## Phase 9 — UI: charts and the segfault

### Overview
Fixes F6 and F9. Makes the product visually presentable and the test suite screenshot-able.

### Changes Required

**1. Fix the crash.**
- `frontend/live_client.py:55`: parent the socket — `QWebSocket(parent=self)`. The `QTimer` is parented and the socket is not, leaving C++ destruction order undefined while signals may be in flight.
- `tests/frontend/server_fixture.py:76`: stop the live loop and monitoring pipeline explicitly *before* signalling uvicorn, and **raise** if the thread fails to join rather than leaking live threads into the next test.
- `api/app.py:87`: guard `asyncio.run_coroutine_threadsafe` against a closing loop and drop the broadcast if the loop is gone.

**2. Live trend chart with the forecast on it.** New `frontend/widgets/live_chart.py`, modelled on the existing QtCharts usage in `frontend/widgets/predicted_vs_actual.py:14`. A rolling window of actual CPU and memory as solid lines, with the current forecast as a **dashed continuation past "now"** and a confidence band. This is the moment that demonstrates "predictive": the solid line arrives where the dashed line said it would.

**3. Contention probability strip.** A slim bar under the chart showing predicted contention probability over the same window with the fitted decision threshold marked — so a viewer can see the probability rise *before* the spike.

**4. Put it on the dashboard.** Insert between the tile row and the diagnosis frame; keep the tiles.

**5. Honest empty and replay states.** Consume `/readiness` from Phase 0 — replace the permanently-green "Live" banner (`dashboard.py:132`) with the real state (`Collecting — 2m15s of 5m`, `No model trained`, `Below skill gate`, `Live`, `REPLAY`).

### Success Criteria
**Automated:**
- [ ] `QT_QPA_PLATFORM=offscreen pytest tests/frontend -q` exits **0** on three consecutive runs
- [ ] Full `pytest -q` completes with no fatal error
- [ ] The chart renders actual and forecast series from a `LiveStateOut` payload
- [ ] Each readiness state renders its own banner text
- [ ] `server_fixture` raises if the server thread fails to join

**Manual:**
- [ ] During replay of a recorded event, the forecast line rises **before** the actual line
- [ ] The contention strip crosses its threshold ahead of the spike
- [ ] A fresh install shows "no model trained", not a green "Live"

---

## Phase 10 — Evidence pack for Review 2

### Overview
Assembles the six required answers from work already done. Nothing new is invented here; this phase is about making the evidence presentable and reproducible.

### Deliverables

**1. Architecture.** A rendered diagram of the pipeline — Monitoring → Ingestion → Storage → Features (system + process) → Forecasting / Contention / Anomaly → Diagnosis → Recommendation → API → Frontend, with the retraining and reporting loops. Committed as a source file, not a one-off image.

**2. Module table.** Each of the eleven modules with its responsibility, source path, test count, and **behavioural vs plumbing** test split. Any module whose completion cannot be backed by a test asserting a *correct answer* (not merely that a function was called) is marked as such. This is deliberate: the project reached this state precisely because 386 green tests asserted plumbing and nothing about quality.

**3. Results.** Generated by `aiperfopt evaluate --report` — the ablation table, PR curve, confusion matrix, baseline comparison, SHAP importance, dataset summary and overhead measurement.

**4. Screenshots.** Dashboard with live chart and forecast; Predicted vs. Actual with a real pairing; Reporting with populated accuracy; Retraining with gate pass/fail; `aiperfopt doctor`; a clean full test run; the diagnosis + recommendation panel during a replayed event.

**5. Demo script.** A written, rehearsed sequence: `doctor` → show readiness → start replay → watch forecast lead the actual → contention probability crosses threshold → diagnosis resolves to an actionable category → recommendation appears → open Predicted vs. Actual to show the prediction preceded the event.

**6. Honest limitations slide.** What is measured, what is not, and what the numbers do and do not support. Included because a review that asks "Results and Analysis" is best answered by someone who can state the boundaries of their own result.

### Success Criteria
**Automated:**
- [ ] `aiperfopt evaluate --report` regenerates every figure and table from the collected data in one command
- [ ] The module table's test counts are generated from the suite, not hand-maintained

**Manual:**
- [ ] Every one of the six Review 2 requirements maps to a concrete artifact
- [ ] The demo has been rehearsed end-to-end at least twice from a cold start

---

## Review 2 Requirement Mapping

| Requirement | Answered by | Evidence |
|---|---|---|
| **1. System Design and Architecture** | Phase 10.1 | Architecture diagram; layered config; single-source-of-truth feature path; storage schema |
| **2. Module Identification** | Phase 10.2 | Eleven-module table with responsibilities and source paths |
| **3. Module Completion** | Phases 3, 4, 6, 7 + 10.2 | Per-module behavioural test counts; all 7 diagnosis categories reachable; recommendation engine producing rows |
| **4. Results and Analysis** | **Phase 5** | Ablation (system-only vs system+process), PR curve, baseline comparison, SHAP importance, overhead budget |
| **5. Screenshots of All Outputs** | Phase 10.4 | Dashboard with forecast overlay, Predicted vs. Actual, Reporting, Retraining gate, CLI, clean test run |
| **6. Live Demonstration** | **Phase 8** + 10.5 | Deterministic replay of a real recorded pressure event, rehearsed script |

---

## Testing Strategy

### The gap being closed
386 tests pass while every model has zero skill, because the suite asserts that functions were called, records written and threads started — never that an answer was correct. Every phase here adds at least one test that fails when *behaviour* degrades, not merely when code errors.

### Test categories added
1. **Train/serve equivalence** — identical input through training and live paths yields identical feature vectors and identical scores. Catches the F1/F4 class entirely.
2. **Causality** — no feature may depend on a future sample.
3. **Distribution** — the detector must not flag its own training data, nor ordinary real data.
4. **Reachability** — every diagnosis category and the recommendation engine must be provably reachable.
5. **Skill** — fixed-seed training must clear the quality gate.
6. **Replay fidelity** — replayed sessions reproduce original feature vectors at any speed.
7. **Lifecycle** — no test may leak a live thread past teardown.
8. **Overhead** — the monitor's own footprint stays within budget.

### Marking
Every test is marked `behavioural` or `plumbing`, with `--strict-markers` enforced. The ratio is published in the README and in the Phase 10 module table, because an unqualified "386 tests pass" is the exact claim that concealed this project's condition.

---

## References

- Process data collected and discarded: `src/aiperfopt/features/config.py:54`; `grep -rn "process" src/aiperfopt/features/` returns nothing
- Existing launch/terminate detector: `src/aiperfopt/monitoring/process_tracker.py`, `ProcessEvent` in `src/aiperfopt/storage/records.py`
- Anomaly imputation skew: `src/aiperfopt/anomaly/detector.py:83`, `:100`, `:111`
- Diagnosis short-circuit: `src/aiperfopt/diagnosis/taxonomy.py:228`
- Excluded `current_value`, duplicated: `src/aiperfopt/forecasting/training_pipeline.py:265`, `src/aiperfopt/forecasting/evaluation.py:32`
- Forecasts never persisted: `src/aiperfopt/api/live_state.py`; unused writer at `src/aiperfopt/storage/writer.py:155`
- Contention labeling thresholds: `src/aiperfopt/contention/config.py`
- Unparented socket: `src/aiperfopt/frontend/live_client.py:55`
- Leaked test threads: `tests/frontend/server_fixture.py:76`
- Chart precedent: `src/aiperfopt/frontend/widgets/predicted_vs_actual.py:14`
- Superseded plan: `docs/plans/2026-08-24-make-it-actually-predict.md`
