# Honest Limitations (Phase 10.6)

What is measured, what is not, and what the numbers do and do not
support — Review 2 asks for "Results and Analysis" from someone who can
state the boundaries of their own result, not just a scoreboard.

## What the evidence pack currently shows

The freshest `aiperfopt evaluate --report` run (11.2 hours of real,
passively-collected accumulated usage, 26,822 rows — see
`results/dataset_summary.txt`) is **honest but weak**, and that gap is
itself the most important limitation to state out loud:

- **The contention classifier did not train.** `results/dataset_summary.txt`:
  "no contention classifier was trained on this window (labels came back
  single-class — reported honestly rather than fabricated)." In ~11 hours
  of ordinary development-machine use, no interval crossed
  `ContentionLabelingConfig`'s threshold for a positive example. This is
  the project's stated headline model (Decisions Taken: "Headline model:
  Contention prediction (classification)"), and its real-data evidence
  pack currently has **zero** PR curve, confusion matrix, or SHAP
  importance plot to show — not because the code is broken, but because
  the input data hasn't contained a real pressure event yet.
- **The ablation shows no measured process-feature gain on this window.**
  `results/ablation.csv`'s `v1` (system-only) and `v2` (system+process)
  arms are numerically identical for every forecast target. Walk-forward
  model selection picked the `moving_average` baseline over LightGBM in
  *both* arms on this holdout, and a baseline that ignores engineered
  features entirely produces identical numbers regardless of what
  features were available to the model that lost. The central thesis of
  this repair plan — that process-launch signal gives LightGBM a real
  edge — is architecturally implemented and unit-tested (see
  `tests/test_features_process_features.py`,
  `tests/test_contention_processes.py`), but **not yet demonstrated on
  real accumulated data**, only on synthetic/replay data.
- **Forecasting still loses to naive persistence on this window**
  (`results/baseline_comparison.csv`: e.g. `cpu_percent_short` LightGBM
  MAE 2.18 vs. persistence MAE 1.28). Expected, not alarming: on a mostly
  idle machine, next-tick CPU really is close to current-tick CPU, and
  persistence is close to the theoretical best predictor for that regime.
  A forecaster only earns its keep once real load variation exists in the
  window it's evaluated against.

- **The currently-deployed forecaster itself is a moving-average baseline,
  not LightGBM — confirmed directly against the deployed artifact, not
  just this evaluation run.** Querying
  `get_evaluations_by_version(engine, "phase7-real-deploy")` on the
  primary database shows the `holdout_real_usage`-tagged row for
  `cpu_percent`/`short` has `model_name == "moving_average"`, MAE 3.22
  (medium horizon: MAE 4.21). This is the concrete, real-data reason a
  live Predicted-vs-Actual chart can show large, spiky prediction error
  around a sharp usage spike: a moving average is structurally unable to
  anticipate a spike before it happens, it only smooths the recent trend.
  It is the same weakness the ablation above shows in aggregate, just
  traced to the specific artifact actually serving `/live/state` right
  now. Anyone asking "why does the forecast look bad" should re-run this
  query against `aiperfopt doctor`'s reported database and active model
  version — it is faster than re-deriving the answer from a fresh
  `evaluate --report` run.

**Why this isn't a regression or a new bug:** the plan's own "Decisions
Taken" table assumes "passive collection over *several days* of normal
use," not eleven hours. The standing background collector has been
running continuously since Phase 2 and keeps accumulating; re-running
`aiperfopt evaluate --report` after more real, varied usage (a build, a
video call, several heavy applications opening) has occurred should start
producing positive contention labels and a real ablation delta. Until
then, the **deployed** model (`phase7-real-deploy`, currently serving
`/live/state` per `aiperfopt doctor`) and the **replay of a recorded real
pressure event** (Phase 8; see `docs/phase10-demo-script.md`) are the
actual evidence for "the model works," not this particular ablation run.

## Other known, deliberate boundaries

- **Confidence is not a calibrated interval.** `ForecastPredictionOut.confidence`
  is a single 0-1 scalar derived from walk-forward RMSE
  (`ForecastModel.set_confidence_from_validation`), explicitly flagged
  "uncalibrated" (ADL-11). The live chart's shaded band
  (`half_width = (1 - confidence) * 15.0` percentage points) is a display
  heuristic that widens as confidence drops — it is not a statistical
  prediction interval and should never be presented as one.
- **The overhead budget is 3.0% CPU, not the originally-stated 2%.**
  `src/aiperfopt/monitoring/overhead.py`'s `CPU_BUDGET_PERCENT = 3.0` is a
  deliberate, already-documented adjustment ("refined against measured
  steady-state overhead on real hardware") from an earlier phase, not a
  new relaxation made to pass this evaluation.
- **macOS only.** Windows CI and verification were explicitly dropped
  from scope by this plan's "Decisions Taken" table — the code stays
  portable by construction, but nothing claims or tests it on Windows.
- **`LiveStateOut` carries no history.** The live dashboard's rolling
  chart accumulates its own client-side buffer per session
  (`deque(maxlen=300)`); there is no server-side "give me the last N
  minutes" endpoint. A fresh page load or app restart starts the chart
  empty even if the backend has hours of persisted history (available via
  `/reporting/summary` and the Predicted vs. Actual tab instead).
- **Two of Phase 9's three manual UI criteria, and Phase 8's REPLAY badge,
  required a real desktop run to verify** (forecast leading actual during
  replay; contention strip crossing threshold ahead of a spike) — not
  achievable from an environment with no way to drive a live Qt window.
  See `docs/phase10-demo-script.md`'s rehearsal log for when these were
  actually verified against a real display.
- **The eleven-module behavioural/plumbing split is a judgment call**,
  not a mechanically derived fact — a human (assisted, per
  `docs/phase10-module-table.md`'s generation script) decided which
  assertions count as checking a *correct answer* versus checking that
  code merely ran. The counts are regenerated from the suite on every
  run; the classification of any individual test is not beyond dispute.

## What *is* solid

- The overhead budget passes on real measurement (2.45% CPU, 3.48 MB/hour
  DB growth), both well inside their ceilings.
- All 7 actionable diagnosis categories and the recommendation engine are
  provably reachable (`tests/test_diagnosis_taxonomy.py`,
  `tests/test_recommendation_service.py`) — the F1/F2 bug class (anomaly
  detector saturating and starving every other diagnosis category) is
  fixed and regression-tested, not just patched once.
- Train/serve feature-vector equivalence, causality (no future leakage),
  and replay fidelity (identical feature vectors at 1x and 10x playback)
  are each backed by a dedicated behavioural test, not merely exercised
  incidentally by a broader integration test.
- The full suite runs clean, headless, with no leaked threads and no
  segfault (a real, previously-unfixed defect this repair effort closed —
  see `docs/plans/2026-09-12-presentable-predictive-monitor.md`'s Phase 9
  "Design notes (crash fix)").
