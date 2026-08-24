# AI System Performance Optimizer — Context Handoff

**Purpose of this document:** paste/upload this into a new chat (along
with the latest code zip) to resume development with zero lost context.
It captures everything that isn't already in `PROJECT_SPEC.md` — i.e. the
decisions made *while implementing* the spec, which a fresh chat has no
way to know.

---

## 1. How to resume in a new chat

1. Upload this file plus the most recent `ai-sysopt-phaseN.zip`.
2. Say something like: *"This is a continuation of the AI System
   Performance Optimizer project. Read PROJECT_HANDOFF.md, then continue
   with Phase N+1."*
3. The workflow rule already in effect: **zips are only generated on
   explicit request** ("give me the zip") — don't expect one after every
   milestone unless asked.

The governing document is still `PROJECT_SPEC.md` (uploaded separately as
a project file) — it is the single source of truth for *requirements*.
This handoff document is the record of *implementation decisions made
while satisfying those requirements*, which is a different and
complementary thing.

---

## 1a. How to actually run it

```bash
# 1. Unzip, then from the project root:
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Run the test suite to confirm everything works in your environment.
pytest -q          # expect: 402 passed, ~11-13 minutes on a modest machine
                   # (test_live_inference_plumbing.py alone is ~225s; tests/frontend/
                   # adds another ~25s for the Qt suite, most of it real
                   # background-thread/websocket/real-server tests, not the ML fitting)

# 3. Run the full ML pipeline (Phases 4-7) — see README.md's "Running the
#    full Phase 0-6 pipeline end-to-end" section for the exact code, or:
python3 -c "
from aiperfopt.bootstrap import bootstrap
from aiperfopt.features import FeatureEngineeringConfig
from aiperfopt.forecasting import train_and_evaluate_all
from aiperfopt.contention import ContentionLabelingConfig, train_contention_and_anomaly_pipeline

ctx = bootstrap()
fconfig = FeatureEngineeringConfig()
forecast_summary = train_and_evaluate_all(
    ctx.storage_engine, fconfig, ctx.settings.forecasting, model_version='v1',
    target_metrics=['cpu_percent', 'mem_percent'],
)
short_outcomes = {o.metric_name: o for o in forecast_summary.outcomes if o.horizon_label == 'short'}
summary = train_contention_and_anomaly_pipeline(
    ctx.storage_engine, fconfig, ctx.settings.forecasting, ctx.settings.contention,
    ContentionLabelingConfig(), short_outcomes, model_version='v1',
)
print(summary.walk_forward_summary, summary.holdout_metrics)
"

# 3b. Run Model Retraining (Phase 8) against accumulated real-usage data
#    (requires the live monitor, or another source, to have already
#    written rows into raw_system_metrics -- there's nothing to retrain
#    on with a freshly bootstrapped/empty database):
python3 -c "
from aiperfopt.bootstrap import bootstrap
from aiperfopt.features import FeatureEngineeringConfig
from aiperfopt.contention import ContentionLabelingConfig
from aiperfopt.retraining import (
    run_retraining_on_real_usage_data, confirm_deployment, get_active_model_version,
)

ctx = bootstrap()
candidate = run_retraining_on_real_usage_data(
    ctx.storage_engine, FeatureEngineeringConfig(), ctx.settings.forecasting,
    ctx.settings.contention, ContentionLabelingConfig(), ctx.settings.retraining,
    new_model_version='v2', target_metrics=['cpu_percent', 'mem_percent'],
)
print(candidate.comparison.recommendation, candidate.comparison.reasons)
# Deployment is never automatic -- a human decides:
confirm_deployment(ctx.storage_engine, ctx.settings.retraining, 'v2', candidate.comparison, confirmed=True)
print('active version:', get_active_model_version(ctx.settings.retraining))
"

# 4. To run the LIVE monitor separately (real psutil data, no ML):
python3 -c "
from aiperfopt.bootstrap import bootstrap
from aiperfopt.pipeline import create_monitoring_pipeline
import time

ctx = bootstrap()
pipeline = create_monitoring_pipeline(ctx)
pipeline.start()
time.sleep(30)
pipeline.stop()
"

# 5. Run the real API server (Phase 9) -- auto-starts Monitoring + live
#    inference on startup (feature -> forecast -> contention -> diagnosis
#    -> recommendation, ticking every api.live_inference_interval_seconds):
python3 -m aiperfopt.api.server
# Then: http://127.0.0.1:8000/docs for OpenAPI docs (FR-12.3),
#       ws://127.0.0.1:8000/ws/live for the live-state WebSocket (FR-12.1).
# Nothing is deployed yet on a fresh database -- POST /retraining/run then
# /retraining/confirm (see Section 3d below) before live inference has a
# model to use; until then /live/state reports metrics/features only.

# 6. Run the PySide6 frontend (Phase 10) -- talks to whatever's at
#    AIPERFOPT_FRONTEND__API_HOST/PORT (default 127.0.0.1:8000); the
#    WebSocket client retries indefinitely, so it's fine to start this
#    before or after step 5:
python3 -m aiperfopt.frontend.main
```

Notes:
- Step 3 takes roughly 1-3 minutes depending on machine speed (Exponential Smoothing fitting is the slow part — see Section 6 below).
- Config overrides use `AIPERFOPT_SECTION__FIELD=value` env vars — see `config/.env.example`; the frontend has its own small env prefix, `AIPERFOPT_FRONTEND__` (host/port only — see Section 3e).
- `pytest -q` is the fastest way to confirm the environment is set up correctly before trying anything else.
- Phase 9 added a real API server (`python3 -m aiperfopt.api.server`) — see Section 3d for the full request flow (retrain -> confirm -> live inference -> WebSocket).
- Phase 10 added a real PySide6 frontend (`python3 -m aiperfopt.frontend.main`) — see Section 3e.
- Phase 11 added `GET /reporting/summary` (daily/weekly/custom-window exportable reports) — see Section 3f.

---

## 2. Current status

| Phase | Subsystem | Status |
|---|---|---|
| 0 | Project Foundation (config, logging, storage bootstrap, exceptions) | ✅ Complete |
| 1 | Data Storage (schema, rollup/retention) | ✅ Complete |
| 2 | Monitoring Engine + Data Ingestion | ✅ Complete |
| 3 | Feature Engineering | ✅ Complete |
| 4 | Forecasting Models + Evaluation Harness | ✅ Complete |
| 5 | Contention Prediction + Anomaly Detection | ✅ Complete |
| 6 | Diagnosis / Explainable AI | ✅ Complete |
| 7 | Recommendation Engine | ✅ Complete |
| 8 | Model Retraining | ✅ Complete |
| 9 | API Layer | ✅ Complete |
| 10 | Frontend | ✅ Complete |
| 11 | Reporting / Analytics | ✅ Complete |

**Every Section 7.1 subsystem is now at least minimally implemented.**
V1 is functionally complete against the subsystem map; what remains
(per `PROJECT_SPEC.md` Section 20) is post-V1 roadmap territory
(statistical calibration, multi-horizon sequence models, cross-platform
support, etc.), not a missing V1 piece. Worth doing a full
spec-compliance pass across every FR/NFR before treating V1 itself as
formally "done," per the previous handoff's own suggestion -- not done
as part of this session, flagged for the next one (Section 8).

**Test suite:** 402 tests, all passing in isolation and in a clean
full-suite run (~11-13 minutes total on a modest machine, no flakes in
the final Phase 11 run). `test_live_inference_plumbing.py` alone
accounts for ~225s of that (several full real LightGBM/contention
training pipelines end-to-end, per the project's "no ML mocking" rule);
`tests/frontend/` (49 tests) adds roughly another ~25s, mostly real
background-thread/real-websocket/real-local-server tests
(`tests/frontend/server_fixture.py` spins up an actual `uvicorn` server
on a real port), not ML fitting. The previous handoff noted two tests
that flake specifically under heavy concurrent CPU load
(`test_retraining_panel.py`'s round-trip test and the pre-existing
`test_real_subprocess_termination_is_detected`) -- neither reproduced in
this session's full-suite run, consistent with that being environment
contention rather than a real bug, but worth remembering if either
reappears.

**Code lives at** `/home/claude/ai-sysopt` inside the conversation
container that built it. That container does not survive a new chat —
only the zip does. Always get a zip before a session might end.

---

## 3. Architecture map (as built)

```
src/aiperfopt/
├── common/exceptions.py        # Shared exception hierarchy, one subclass per subsystem
├── config/
│   ├── settings.py              # AppSettings (pydantic-settings), layered: env > .env > user yaml > default yaml
│   └── __init__.py
├── applog/
│   ├── formatting.py             # JsonFormatter (one JSON object per log line)
│   └── setup.py                  # configure_logging() / get_logger() / reset_logging_for_tests()
├── storage/
│   ├── engine.py                  # SQLAlchemy Core engine + SQLite pragmas (WAL, foreign_keys)
│   ├── schema.py                   # All 9 tables (Table objects, shared MetaData)
│   ├── schema_management.py        # create_all_tables() / verify_schema()
│   ├── records.py                  # Typed dataclasses: SystemMetricSample, ProcessMetricSample, etc.
│   ├── writer.py                    # Single + batch insert functions, one per table
│   ├── reader.py                    # Chronological range-query functions, one per table
│   └── rollup.py                    # rollup_system_metrics(), prune_process_level_data(), apply_retention_policy()
├── monitoring/                  # Producer side (ADL-18 thread+queue pattern)
│   ├── collectors.py              # collect_system_metrics() — per-source graceful degradation
│   ├── process_tracker.py         # ProcessTracker — pid->Process cache, launch/terminate diffing
│   ├── queue_items.py             # QueueItem/QueueItemKind — tagged union on the shared queue
│   └── engine.py                    # MonitoringEngine — 2 threads (system metrics, process metrics)
├── ingestion/                   # Consumer side
│   └── consumer.py                # IngestionConsumer — 1 thread, batches + flushes + retention trigger
├── features/                    # Feature Engineering
│   ├── config.py                   # FeatureEngineeringConfig — windows, tolerances, metric lists
│   ├── engine.py                    # compute_feature_matrix() — THE single source of truth (FR-4.2)
│   └── storage_bridge.py            # compute_live_feature_vector(), backfill_feature_matrix()
├── forecasting/                 # Forecasting Models + Evaluation Harness
│   ├── synthetic.py                 # Scenario generator (idle, cpu_stress, mem_pressure, disk_burst, benign_spike)
│   ├── dataset.py                    # build_supervised_dataset() — joins features with forward-looking labels
│   ├── baselines.py                   # Forecaster protocol + naive/moving_avg/exp_smoothing baselines
│   ├── model.py                        # MetricForecaster — LightGBM wrapper, one per (metric, horizon)
│   ├── persistence.py                  # save_forecaster() / load_forecaster() — disk artifacts, not DB rows
│   ├── evaluation.py                    # run_walk_forward_evaluation(), select_best_forecaster_name()
│   └── training_pipeline.py             # train_and_evaluate_all() — the Phase 4 entrypoint
├── pipeline.py                  # create_monitoring_pipeline() — wires Monitoring+Ingestion; NOT auto-started
└── bootstrap.py                  # bootstrap() — config -> logging -> storage(+schema); the one required entrypoint

tests/            # One test module per subsystem, mirrors src/ layout exactly
config/
├── default.yaml    # Packaged defaults (lowest-priority config layer)
└── .env.example     # Documents every AIPERFOPT_* env override
```

**Phase 5 additions** (not yet reflected in the tree above — added here rather than re-drawing it):

```
src/aiperfopt/
├── synthetic.py                 # PROMOTED from forecasting/synthetic.py — now shared by Forecasting/Contention/Anomaly
├── anomaly/
│   ├── detector.py                # AnomalyDetector — Isolation Forest, normalized [0,1] score
│   ├── persistence.py              # save/load via joblib (sklearn has no native format)
│   └── notifications.py             # emit_anomaly_notification_if_significant() -> app_logs
├── contention/
│   ├── config.py                   # ContentionLabelingConfig — training-label thresholds (plain dataclass, like FeatureEngineeringConfig)
│   ├── dataset.py                   # build_contention_dataset() — forecast+state+anomaly join (FR-6.1)
│   ├── model.py                      # ContentionClassifier — logistic regression + coefficient attribution
│   ├── processes.py                   # get_contributing_processes() — lookup against raw_process_metrics
│   ├── prediction.py                   # predict_contention() — live inference, anomaly dampening, full record
│   ├── persistence.py                   # save/load via joblib
│   ├── evaluation.py                     # walk-forward precision/recall/F1
│   └── training_pipeline.py               # train_contention_and_anomaly_pipeline() — the Phase 5 entrypoint
└── forecasting/
    └── deployment.py                # load_deployed_forecaster() — new: reconstruct whichever forecaster won
```

**Phase 6 additions:**

```
src/aiperfopt/
├── storage/
│   ├── schema.py                    # +diagnoses table
│   ├── records.py                    # +DiagnosisRecord
│   ├── writer.py                      # +insert_diagnosis()
│   └── reader.py                       # +get_diagnoses_range()
├── config/
│   └── settings.py                  # +DiagnosisSettings (cross_resource_dominance_ratio,
│                                     #  process_dominance_ratio, memory_sudden_rate_of_change_threshold, shap_top_n)
└── diagnosis/
    ├── taxonomy.py                   # DiagnosisCategory (Section 12's 8 values) + resolve_diagnosis_category()
    ├── explainer.py                    # Tiered lightweight/SHAP forecast attribution; should_use_shap_tier()
    ├── text.py                          # Technical (Developer Mode) + plain-language (General Mode) templates
    └── service.py                        # diagnose_contention_event() — the Phase 6 entrypoint
```

**Phase 7 additions:**

```
src/aiperfopt/
├── storage/
│   ├── schema.py                    # +recommendations table
│   ├── records.py                    # +RecommendationRecord
│   ├── writer.py                      # +insert_recommendation()
│   └── reader.py                       # +get_recommendations_range()
└── recommendation/
    ├── catalog.py                 # RecommendationCatalog + load_recommendation_catalog()
    └── service.py                  # generate_recommendations() — the Phase 7 entrypoint

config/
└── recommendations.yaml   # The curated category -> actions mapping (NFR-12.1) — the
                            # ENTIRE decision content of the Recommendation Engine lives
                            # here, not in Python source. Edit this file to change wording
                            # or actions; bump catalog_version when you do (FR-9.3 traces
                            # every persisted recommendation back to which catalog revision
                            # produced it).
```

**Phase 8 additions:**

```
src/aiperfopt/
├── storage/
│   ├── schema.py                    # +model_deployments table
│   ├── records.py                    # +ModelDeploymentRecord
│   ├── writer.py                      # +insert_model_deployment()
│   └── reader.py                       # +get_model_deployments_range(), get_latest_model_deployment(),
│                                       #  get_evaluations_by_version() (finds a version's own past holdout rows)
├── config/
│   └── settings.py                  # +RetrainingSettings (active_version_file, regression_tolerance,
│                                     #  real_usage_holdout_fraction)
├── forecasting/
│   └── training_pipeline.py         # train_and_evaluate_all()/train_evaluate_and_select_one() gained
│                                     #  optional scenarios/holdout_samples/holdout_dataset_tag params
├── contention/
│   └── training_pipeline.py         # train_contention_and_anomaly_pipeline() gained the same params
└── retraining/
    ├── real_usage_data.py           # build_real_usage_training_data() — chronological train/holdout split
    │                                 #  of accumulated raw_system_metrics history
    ├── active_version.py            # read_active_version()/write_active_version() — atomic pointer file
    ├── comparison.py                # compare_candidate_to_active() — RetrainingComparison, regression detection
    ├── deployment.py                # confirm_deployment(), rollback_to_last_known_good(),
    │                                 #  get_active_model_version()
    └── pipeline.py                  # run_retraining_on_real_usage_data() — the Phase 8 entrypoint;
                                      #  run_retraining_on_synthetic_data() for exercising it pre-real-data
```

**Key entrypoints for any future code to reuse:**
- `aiperfopt.bootstrap.bootstrap(user_config_path=None) -> AppContext` — always call this first. Does NOT start any threads.
- `aiperfopt.pipeline.create_monitoring_pipeline(ctx) -> MonitoringPipeline` — call `.start()`/`.stop()` to run the live monitor.
- `aiperfopt.features.compute_and_store_live_feature_vector(engine, config)` — the live-inference feature call Forecasting/Contention should call each cycle.
- `aiperfopt.retraining.run_retraining_on_real_usage_data(...)` — the Phase 8 entrypoint; produces a `RetrainingCandidate` for a human to review.
- `aiperfopt.retraining.confirm_deployment(...)` — the only function in this codebase that ever activates a model_version; requires an explicit `confirmed: bool`.
- `aiperfopt.retraining.get_active_model_version(retraining_settings)` — what the live pipeline (Phase 9) should read to know which model_version to load.
- `aiperfopt.features.backfill_feature_matrix(engine, config, start, end)` — the batch call Forecasting should use to build a training set.
- `aiperfopt.forecasting.train_and_evaluate_all(engine, feature_config, forecasting_settings, model_version, target_metrics=None)` — trains every metric/horizon, evaluates against baselines, persists results, saves winning LightGBM artifacts. This is what a future retraining trigger (Phase 8) should call.
- `aiperfopt.forecasting.load_deployed_forecaster(outcome, models_dir, model_version) -> Forecaster` — the correct way to get a ready-to-use forecaster (LightGBM or a baseline) from a training outcome. Use this, not `load_forecaster` directly, unless you already know LightGBM won.
- `aiperfopt.contention.train_contention_and_anomaly_pipeline(engine, feature_config, forecasting_settings, contention_settings, labeling_config, forecast_outcomes, model_version)` — the Phase 5 entrypoint. Requires `forecast_outcomes` (short-horizon only) from an already-completed Phase 4 run.
- `aiperfopt.contention.predict_contention(classifier, engine, feature_row, as_of, decision_threshold, anomaly_dampening_factor, contributing_processes_top_n, model_version)` — the live-inference call that assembles a full `ContentionPredictionRecord`. Not yet wired into a running loop (same "Phase N builds the pieces, a later phase with an actual consumer wires the loop" pattern as Phase 3).
- `aiperfopt.diagnosis.diagnose_contention_event(engine, contention_record, feature_row, deployed_forecasters, contention_settings, diagnosis_settings, model_version, explicit_request=False) -> DiagnosisRecord | None` — the Phase 6 entrypoint. Takes the `ContentionPredictionRecord` `predict_contention` already produced (does *not* need the live `ContentionClassifier` — see `diagnosis/service.py`'s docstring for why) plus the same-cycle full engineered feature vector (for SHAP attribution and the memory gradual/sudden split) and the `deployed_forecasters` mapping (same one Phase 5's training pipeline builds via `load_deployed_forecaster`). Returns `None` and persists nothing when there's nothing to diagnose (not triggered, no significant anomaly) — see "Design decisions" below for why that's deliberate, not a gap. The returned record's `.id` is populated (see bug #10 below) — it's needed by Phase 7.
- `aiperfopt.recommendation.load_recommendation_catalog(path=None) -> RecommendationCatalog` — loads and eagerly validates `config/recommendations.yaml` (defaults to the packaged file). Call this once (e.g. at startup, alongside `bootstrap()`) and reuse the returned catalog — no need to reload per event, the file doesn't change at runtime.
- `aiperfopt.recommendation.generate_recommendations(engine, diagnosis, catalog) -> RecommendationRecord | None` — the Phase 7 entrypoint. Takes the `DiagnosisRecord` `diagnose_contention_event` already produced and persisted (its `.id` must be set — raises `ValueError` otherwise, since FR-9.3 traceability has nothing to point to without it). Returns `None` and persists nothing when `diagnosis.is_actionable` is `False` (currently only the anomaly category, ADL-36).
- `aiperfopt.forecasting.resolve_deployed_forecasters(engine, feature_config, forecasting_settings, model_version, horizon_label="short", target_metrics=None) -> dict[str, Forecaster]` — Phase 9's addition: reconstructs every deployed forecaster for a model_version purely from persisted evaluation history, no in-memory `TrainingRunSummary` required. Use this (not `load_deployed_forecaster` directly) any time you have a `model_version` string but not the training run that produced it — which is the normal case for anything running in a different process than training (the live-inference loop, first).
- `aiperfopt.contention.build_live_contention_input_row(samples, feature_vector, deployed_forecasters, anomaly_detector, feature_config) -> pd.Series | None` — the live-inference counterpart to `build_contention_dataset`: builds exactly one `forecast_{metric}`/`current_{metric}`/`anomaly_score` row for *now*, ready to pass straight into `predict_contention`.
- `aiperfopt.api.create_app(ctx=None, user_config_path=None, auto_start=False) -> FastAPI` — the Phase 9 entrypoint. Builds but does not start Monitoring/live-inference unless `auto_start=True` (the real server passes this; a test building an app for `TestClient` normally shouldn't). `python3 -m aiperfopt.api.server` is the real server process.
- `aiperfopt.api.LiveInferenceLoop` — the running feature→forecast→contention→diagnosis→recommendation cycle (Phase 9's core addition). `.start()`/`.stop()` like `MonitoringEngine`; `.latest_state()` returns the most recent `LiveState` snapshot (or `None` before the first cycle completes).

---

## 3a. Phase 6 addendum: what was actually built

Phase 6 turned out to have one genuine design question the spec doesn't
fully resolve on its own — how "tiered" (Section 10.5) applies
differently to the two things Diagnosis explains:

- **Forecasting attribution is genuinely tiered.** A LightGBM forecaster
  has two real options with a real cost difference: native
  `booster.feature_importance()` (free, global, lightweight) vs.
  `shap.TreeExplainer(...).shap_values(row)` (a real per-request
  computation — the actual "SHAP-class method" Section 10.5 means).
  `diagnosis/explainer.py` implements both and switches on
  `should_use_shap_tier()`'s three triggers (high-confidence contention,
  significant anomaly, explicit request), in that priority order.
- **Contention attribution is *not* meaningfully tiered — and this is
  proven, not just claimed.** `contention/model.py`'s own docstring
  already explains why logistic regression was chosen over a small GBM:
  its coefficients give free, exact per-feature attribution. It turns out
  that attribution *is* SHAP's linear decomposition for a standardized
  linear model (`shap.LinearExplainer` with a zero/mean background
  reduces to exactly `coefficient * standardized_value`, the same formula
  `ContentionClassifier.contributing_metrics()` already computes). This
  isn't asserted in a docstring and left to rot — `test_diagnosis_explainer.py::test_contention_coefficient_attribution_equals_real_shap_linear_explainer`
  checks it numerically against a real `shap.LinearExplainer` call
  (matches to 1e-9). Consequently `diagnosis/service.py` never recomputes
  contention attribution through SHAP — it reuses
  `ContentionPredictionRecord.contributing_metrics` directly regardless
  of tier, and `explainer.py` deliberately has no `explain_contention`
  function so there's no implied second code path.
- **Which metrics get a forecast explanation depends on the resolved
  category**, not "explain every deployed forecaster every time"
  (`service.py`'s `_metrics_to_explain`): a CPU-dominant diagnosis
  explains `cpu_percent`/`load_avg_1` only, Cross-Resource Contention
  explains every deployed metric (no single one is more relevant by
  definition of that category), and the anomaly category explains none
  (an anomaly score is a feature-vector-distance signal, not a
  per-metric forecast question).
- **A real bug was caught and fixed during development, not just during
  review**, via a manual end-to-end smoke test before the formal test
  suite was written: the plain-language template for CPU-Bound *Single*
  Process originally named every process in `contributing_processes`
  (top-3), producing text like *"stress_test and idle_proc is using a
  large share of your CPU"* — directly contradicting the category's own
  claim that one process dominates, and grammatically broken to boot.
  Fixed in `diagnosis/text.py` to use only the top process's name for
  that one category; every other category's template correctly uses the
  plural list. Regression test:
  `test_diagnosis_text.py::test_cpu_single_process_names_only_the_dominant_process`.
- **Diagnosis only produces output for events actually worth explaining**
  — not every monitoring cycle. `resolve_diagnosis_category()` returns
  `None` when contention wasn't triggered and there's no significant
  anomaly (the overwhelming majority of cycles), and
  `diagnose_contention_event()` propagates that `None` straight through
  rather than manufacturing a "system is healthy" diagnosis record.
  Documented as a deliberate reading of FR-8.3 ("every prediction event
  resolves to exactly one category") scoped to actual triggered/flagged
  events, not routine background cycles — see `taxonomy.py`'s module
  docstring, point 2.
- **`ContentionSettings.anomaly_significant_threshold` is reused, not
  duplicated**, as the "significant anomaly" trigger for both the Section
  12 taxonomy category *and* Section 10.5's SHAP-tier trigger. A second
  `DiagnosisSettings` copy of that number would be exactly the kind of
  config-drift risk this codebase has otherwise avoided (see bug #4's
  `epoch_seconds_to_utc_datetime()` precedent) — callers pass the one
  canonical value in explicitly at each call site.

---

## 3b. Phase 7 addendum: what was actually built

Phase 7 is deliberately the thinnest subsystem in this codebase, and
that's by design, not underdevelopment — FR-9.1/ADL-15/ADL-19 all point
the same direction: consume only the standardized diagnosis category,
be rule-based, and never reach past Diagnosis to raw metrics or
Contention Prediction directly. What actually got built:

- **The entire decision surface is one YAML file**
  (`config/recommendations.yaml`), not Python source — consistent with
  NFR-12.1 ("all operational parameters externally configurable without
  modifying application source code") and, unlike `DiagnosisSettings`'
  numeric thresholds, recommendation *wording* is exactly the kind of
  content a non-developer (or a later version of you) would want to edit
  without touching code. `recommendation/catalog.py`'s loader validates
  the file eagerly at load time — every actionable Section 12 category
  must be present, the anomaly category must be absent, every action
  needs all three of title/description/rationale — so a bad edit to the
  YAML fails loudly at startup with a specific `RecommendationError`
  rather than surfacing as a confusing `None` or `KeyError` deep in a
  live diagnosis cycle.
- **Two independent enforcement points for ADL-36**, not one:
  `recommendation/service.py`'s `generate_recommendations()` checks
  `diagnosis.is_actionable` before ever touching the catalog, *and*
  `RecommendationCatalog.actions_for()` itself refuses a lookup for the
  anomaly category even if called directly, bypassing the service layer
  entirely. Belt-and-suspenders deliberately: FR-9.2's "never take
  automated action... always user-confirmed" is a hard constraint the
  spec treats as non-negotiable, so it gets two independent checks
  instead of trusting a single call site to always remember the gate.
- **Recommendation content is deliberately generic, not
  per-instance-specific** — no process names, no numeric thresholds, no
  reference to a specific event's forecast values. That specificity
  already lives in the Diagnosis explanation (FR-8.1/FR-8.2) a person
  reads right alongside the recommendation; duplicating it here would
  both repeat information and, more importantly, blur the line FR-9.1
  draws between "Diagnosis explains what's happening" (instance-specific)
  and "Recommendation Engine maps category to curated action"
  (category-level only, by construction of the lookup itself — there is
  no code path by which a raw metric value *could* influence which
  action text gets returned).
- **`RecommendationRecord` denormalizes `diagnosis_category` alongside
  `diagnosis_id`**, mirroring how `contention_predictions` already
  stores its own `contributing_metrics`/`contributing_processes` rather
  than requiring a join back to `forecast_predictions` — a future
  Reporting or Frontend query filtering recommendations by category
  shouldn't need to join back to `diagnoses` just to do that.
- **No `RecommendationSettings` was added to `AppSettings`** — there's no
  genuine runtime-tunable numeric parameter here (unlike
  `DiagnosisSettings`' dominance ratios), just a file path with a sane
  packaged default, so adding a settings class for it would have been
  ceremony without a real knob behind it. Section 3's "simpler mechanism
  preferred" applied literally: `load_recommendation_catalog(path=None)`
  already accepts an override path directly for whoever eventually wires
  a live pipeline (Phase 9, most likely) and wants to point it elsewhere.

---

## 3c. Phase 8 addendum: what was actually built

Phase 8 deliberately contributes **no new training logic** — that would
duplicate already-tested Phase 4/5 machinery for no behavioral
difference. What it adds is: a real-usage data source, a
compare-against-what's-active step, and the one place a model_version is
ever activated. Concretely:

- **`forecasting.train_and_evaluate_all()` and
  `contention.train_contention_and_anomaly_pipeline()` both gained
  optional `scenarios`/`holdout_samples` parameters.** When omitted,
  behavior is byte-for-byte identical to Phase 4/5 (synthetic generation,
  same defaults) — every existing Phase 4/5 test still passes unchanged.
  When supplied, the exact same walk-forward validation, held-out
  generalization check, artifact saving, and evaluation persistence runs
  against whatever data was handed in. This is the only change to
  already-shipped Phase 4/5 code; everything else about Phase 8 is new
  files under `aiperfopt/retraining/`.
- **`aiperfopt/retraining/real_usage_data.py`** turns accumulated
  `raw_system_metrics` history into that same `scenarios`/
  `holdout_samples` shape via one **chronological** split — the most
  recent `real_usage_holdout_fraction` of rows becomes the holdout,
  never an earlier slice, mirroring Section 11/ADL-10's anti-leakage
  stance for real (not synthetic-scenario-block) data. Deliberately
  *not* a multi-scenario library like Phase 4's synthetic bootstrap:
  real usage has no artificial scenario boundaries, it's one continuous
  recording, so there's exactly one training "scenario"
  (`REAL_USAGE_SCENARIO_NAME = "real_usage"`) plus one holdout slice.
- **`aiperfopt/retraining/comparison.py`** builds a
  `RetrainingComparison`: for each (metric, horizon) the candidate
  trained, it looks up the currently-active version's own previously
  recorded holdout evaluation (`get_evaluations_by_version()`, new in
  Phase 8) and flags a regression if holdout MAE or directional accuracy
  moved beyond `regression_tolerance` in the wrong direction; same
  treatment for the contention classifier's holdout F1. **A regression
  on any one comparable outcome blocks the "deploy" recommendation** even
  if every other outcome improved — deliberately conservative, matching
  Section 11's precision-favoring, trust-first stance (ADL-8): a
  candidate that's better on nine metrics and silently worse on one is
  not an unambiguous upgrade. Outcomes with no active-version baseline to
  compare against (e.g. a metric the active version never trained) are
  reported but never block deployment — there's nothing to regress
  relative to. No active version at all → `"no_active_baseline"`,
  meaning "nothing to compare against," not "safe to auto-deploy" — a
  first-ever deployment still requires the same explicit confirmation as
  any other (ADL-23).
- **The recommendation is advisory only — the comparison module never
  activates anything.** `aiperfopt/retraining/active_version.py` (an
  atomically-written pointer file, POSIX `Path.replace` swap so a crash
  mid-write can never leave a corrupted or half-written pointer for the
  live inference path to read) is only ever written by
  `aiperfopt/retraining/deployment.py`'s `confirm_deployment(...,
  confirmed: bool, ...)` — a required, explicit boolean with no default,
  so there is no code path in this codebase that activates a
  model_version without a caller passing `confirmed=True` (FR-11.2). A
  human confirming a "reject"-recommended candidate, or declining a
  "deploy"-recommended one, is recorded faithfully either way — the
  function never overrides the caller's decision with the
  recommendation.
- **Every deployment decision — accepted or rejected — is persisted**
  to a new `model_deployments` storage table (candidate/previous
  version, decision, the recommendation that preceded it, who confirmed,
  and the full serialized comparison for audit). This is the durable
  history `rollback_to_last_known_good()` (NFR-5.3) walks backwards
  through to find the most recent version whose decision was
  `"deployed"` (skipping any rejected candidates in between) — not the
  pointer file, which only ever holds the single current fact.
- **One model_version spans forecasting + contention + anomaly
  together, never independently** — consistent with how Phase 4/5
  already save all three under the same `model_version` string.
  `run_retraining_on_real_usage_data()`/`run_retraining_on_synthetic_data()`
  train and compare all three as one unit; there's no supported path to
  activate a new forecaster paired with an old contention classifier
  (which was trained against a different forecaster's output
  distribution) — that's not a state this system reasons about.
- **`run_retraining_on_synthetic_data()` exists alongside the real-usage
  entrypoint**, even though real-usage retraining is Phase 8's actual
  target per Section 10.3 Phase 2/3. Not scope creep: it's the same
  orchestration code path with a different data source plugged in (a few
  lines), and it lets the comparison/confirmation/rollback machinery be
  exercised and demonstrated before real-usage data has accumulated
  enough to retrain on — directly serving R-3's mitigation ("V1 does not
  depend on Phase 2 data maturing").

---

## 3d. Phase 9 addendum: what was actually built

**Project layout, Phase 9 additions:**

```
src/aiperfopt/
├── forecasting/
│   └── deployment.py                # +resolve_deployed_forecasters() — reconstructs a version's
│                                     #  forecasters from persisted evaluation history alone
├── contention/
│   └── dataset.py                    # +build_live_contention_input_row() — live-inference counterpart
│                                     #  to build_contention_dataset(), for exactly one "now" row
├── config/
│   └── settings.py                  # +ApiSettings (host, port, live_inference_interval_seconds)
└── api/
    ├── app.py                       # create_app() factory; lifespan wires shutdown, never auto-starts
    ├── server.py                     # python -m aiperfopt.api.server — the real uvicorn entrypoint
    ├── live_state.py                  # LiveInferenceLoop + LiveState — the live pipeline itself
    ├── connection_manager.py           # WebSocket broadcast fan-out
    ├── dependencies.py                  # FastAPI Depends() providers, reaching AppContext via app.state
    ├── schemas.py                        # Pydantic response/request models (1:1 with storage records)
    └── routers/
        ├── system.py                     # /health, /config, /monitoring/*, /live/status|start|stop
        ├── history.py                     # GET wrappers over every storage.reader function (FR-12.2)
        ├── retraining.py                   # /retraining/run|confirm|rollback|active-version (Phase 8 over REST)
        └── live.py                          # GET /live/state, WS /ws/live (FR-12.1)
```

Phase 9 is the first phase that had to build genuinely new orchestration
logic, not just wrap an already-complete subsystem — Phases 3 through 8
were each self-contained and individually callable; nothing before this
phase actually ran the pipeline continuously. Concretely:

- **The live-inference loop (`api/live_state.py`) is the actual
  substance of Phase 9** — the API routes are comparatively thin wrappers
  over Phases 1-8's existing functions, but nothing before this phase
  ever ran feature → forecast → contention → diagnosis → recommendation
  on a cadence. It's a dedicated thread (`ADL-18`'s pattern, same as
  Monitoring), not asyncio — every library call it makes (SQLite reads,
  LightGBM/statsmodels inference, scikit-learn, SHAP) is synchronous, and
  FastAPI's async layer only ever reads the loop's latest published
  snapshot, never blocks on it.
- **Two real gaps had to be filled to make the loop possible at all**,
  not just API glue:
  - `forecasting.resolve_deployed_forecasters()` — before Phase 9,
    "reconstruct which forecaster a training run selected" only worked
    if you still had the in-memory `TrainingRunSummary` that run
    produced (`MetricHorizonOutcome.selected_forecaster`). The live loop
    runs in a different process/moment than training, with only a
    `model_version` string to go on. This function rebuilds the needed
    outcome metadata from what's actually persisted: `horizon_seconds`/
    `is_counter` are deterministic from config, and
    `selected_forecaster` is recovered from the holdout evaluation row
    Phase 4/8 already write (via `storage.get_evaluations_by_version`).
  - `contention.build_live_contention_input_row()` — `build_contention_dataset`
    (Phase 5) is training-shaped: it needs a forward-looking label, which
    doesn't exist for "right now." This is the live-inference
    equivalent — one row, no label, same column construction
    (`forecast_{metric}`, `current_{metric}`, `anomaly_score`) so
    `predict_contention` sees exactly the shape it was trained against.
- **Every stage of `_tick()` can legitimately produce nothing, and none
  of that is an error** (NFR-11.1/11.2): not enough trailing history for
  a feature vector; no model deployed yet (Phase 8 never confirmed one);
  a target metric with no current raw value this cycle; a cycle whose
  forecast simply didn't cross the contention threshold. `LiveState`'s
  fields are progressively optional in exactly that order, and a client
  reading `/live/state`/the WebSocket should treat `None` the same way —
  "nothing to show for this stage yet," not a failure. Only a genuine
  problem (a model that failed to load) sets `state.error`.
- **NFR-5.3's automatic fallback is wired at the one place it actually
  needs to be**: `_load_active_models()` catches a load failure for the
  active model_version, calls Phase 8's
  `rollback_to_last_known_good(...)`, and reports what happened via
  `state.error` — never raises out of a cycle. If there's no known-good
  version to roll back to either, the loop still doesn't die; it just
  reports that plainly and tries again next cycle (the active-version
  pointer file itself is untouched by a failed rollback attempt, so nothing
  is left in a half-updated state).
- **The retraining confirmation flow is deliberately stateless across
  requests.** `POST /retraining/run` returns a full `RetrainingComparisonOut`;
  `POST /retraining/confirm` requires the *client* to send that exact
  comparison back, rather than the server re-deriving or re-fetching it
  server-side. This means what a human confirms is provably the same
  evidence `/retraining/run` showed them — the server can't silently
  substitute a fresher (and possibly different) comparison in the gap
  between a human reviewing the result and clicking confirm. A thin
  `_comparison_from_request`/`_comparison_out` pair in
  `routers/retraining.py` handles the round-trip.
- **Nothing starts automatically on `create_app()`** — consistent with
  `bootstrap()` and `create_monitoring_pipeline()` never spawning threads
  as a side effect of being called. A test building an app for
  `TestClient` gets a fully wired-but-dormant app; the real server
  (`api/server.py`) is the one caller that passes `auto_start=True`, so
  Monitoring and live inference are running from the moment the server
  process is actually up (matching the flagship demo's expectation of a
  continuously-running monitor, Section 6).
- **`/config`'s exact JSON key set will need extra thought once the
  Phase 10 frontend exists.** Right now it dumps `ctx.settings` verbatim
  (`model_dump(mode="json")`) — there's nothing sensitive to redact yet
  (Section 13: no credentials/secrets exist in V1), but it's a read-only
  diagnostic view today with no corresponding write endpoint. If a
  future phase adds config *editing* over the API, that's a materially
  different, higher-stakes endpoint than this one and deserves its own
  design pass, not an incidental extension of `/config`.

---

## 3e. Phase 10 addendum: what was actually built

**Project layout, Phase 10 additions:**

```
src/aiperfopt/
├── api/routers/
│   └── reporting.py                # +GET /reporting/predicted-vs-actual (small backend addition)
├── api/schemas.py                  # +PredictedVsActualPointOut
└── frontend/                       # New package -- pure API client (FR-13.2)
    ├── config.py                     # FrontendSettings (host/port/timeout; the one local setting)
    ├── api_client.py                  # ApiClient -- sync httpx wrapper over every REST endpoint
    ├── live_client.py                  # LiveWebSocketClient -- QWebSocket streaming of /ws/live
    ├── workers.py                       # run_in_background() -- QThreadPool helper for slow calls
    ├── formatting.py                     # Pure display-formatting helpers (Qt-free, unit-testable)
    ├── main_window.py                     # MainWindow -- mode toggle, tab assembly, control buttons
    ├── main.py                             # python -m aiperfopt.frontend.main entrypoint
    └── widgets/
        ├── dashboard.py                       # General Mode
        ├── developer_panel.py                  # Developer Mode
        ├── predicted_vs_actual.py               # QtCharts predicted-vs-actual series (FR-13.3)
        ├── predicted_vs_actual_tab.py             # Metric/lookback controls driving the chart
        └── retraining_panel.py                     # Run/confirm/reject/rollback controls
```

**The one backend addition, and why it's backend and not Qt-side.**
PROJECT_HANDOFF.md's own Section 8 (previous "next milestone" writeup)
flagged this explicitly: the predicted-vs-actual join needs both
`/forecasts` and `/metrics/system` for the same window, and joining them
client-side would duplicate logic Section 7.1 item 14's future Reporting
subsystem will also need. `api/routers/reporting.py`'s
`GET /reporting/predicted-vs-actual` does a nearest-timestamp join
(`bisect` over pre-sorted actual samples, 10-second match tolerance --
roughly 5x the recommended 1-2s polling interval, generous enough for
normal jitter without matching across a real monitoring gap) entirely
server-side, using only already-tested `storage.reader` range functions.
A prediction with no actual sample close enough to its `target_timestamp`
(e.g. it targets a moment slightly in the future) comes back with
`actual_value: null`, which the chart widget renders as "not yet plotted
for this point" rather than an error — this was a deliberate design
choice made once, in one place, rather than something every future
consumer of this join has to reimplement.

**Why the frontend is a real Qt application, not a thin demo shell.**
Every widget takes real `LiveStateOut`-shaped JSON and renders real
fields from it — nothing is a placeholder for "would be wired up later."
Concretely:

- **`ApiClient` and `LiveWebSocketClient` are deliberately separate,
  Qt-coupling-wise.** `ApiClient` has zero PySide6 imports and is fully
  testable as plain Python (`tests/frontend/test_api_client.py` uses no
  `qtbot`). `LiveWebSocketClient` is the only place `QObject`/Qt signals
  enter the picture, and it uses `QWebSocket` directly rather than a
  second background thread pushing through a queue -- `QWebSocket`
  already lives inside the Qt event loop, so there was no
  producer/consumer synchronization problem to solve the way
  Monitoring's ADL-18 thread+queue pattern was solving one.
- **Testing this required a real running backend, not an in-process
  shortcut.** `httpx.ASGITransport` (what would let a *sync* `httpx.Client`
  hit a FastAPI app without a real socket) is async-only under the hood
  in a way that doesn't compose with `httpx.Client`'s synchronous API the
  way `fastapi.testclient.TestClient` fakes around internally with an
  `anyio` portal. Rather than fight that, `tests/frontend/server_fixture.py`
  spins up a real `uvicorn.Server` on a background thread bound to a real
  free local port (`socket.bind(("127.0.0.1", 0))` to pick one), and every
  frontend integration test talks to that over real HTTP/WebSocket. This
  is more representative of what the shipped frontend actually does
  anyway, and follows the same "real threads, no mocking" precedent the
  Monitoring/Ingestion tests set in Phase 2.
- **`/retraining/run`'s blocking-for-minutes behavior (flagged as an
  open question in the previous handoff's Section 8) is handled with a
  generic background-job helper (`workers.run_in_background`), not a
  retraining-specific one.** Every `ApiClient` call `MainWindow`/
  `RetrainingPanel`/`PredictedVsActualTab` makes goes through the same
  `QThreadPool`-backed helper -- there's exactly one pattern for "a REST
  call shouldn't block the UI thread," used uniformly, rather than a
  special case only for the one endpoint known in advance to be slow.
  `ApiClient.run_retraining()` additionally overrides its per-request
  `httpx` timeout to 600s (the shared client's default is 5s, sized for
  ordinary fast local reads) since the default would otherwise fire a
  spurious timeout partway through a real multi-minute training run
  even though the request is already safely off the UI thread.
- **General/Developer mode really is just a rendering switch, confirmed
  by construction, not just by convention.** `MainWindow._on_live_state()`
  calls `dashboard.update_state(state)` and
  `developer_panel.update_state(state)` unconditionally, every broadcast,
  regardless of which one is currently visible -- there's no
  mode-conditional fetch or separate "detailed" request path to keep in
  sync with the toggle. Switching modes only ever calls
  `setVisible()`/`setTabVisible()`.
- **`FrontendSettings` (host/port) is the one piece of frontend
  configuration that could not come from `GET /config`**, for the
  obvious bootstrap reason: you need an address before you can make the
  first request to learn anything else. Everything else the frontend
  might have wanted to read locally (e.g. `mode.active_mode`) is
  available from the backend's own `/config` response instead of a
  second, potentially-divergent local copy -- though `MainWindow`
  currently just defaults to General Mode on startup rather than reading
  that value; wiring the initial toggle position to the backend's
  configured default would be a reasonable small follow-up, not a gap
  that was missed by oversight.

---

## 3f. Phase 11 addendum: what was actually built

**Project layout, Phase 11 additions:**

```
src/aiperfopt/
├── reporting/                       # New subsystem package (Section 7.1 item 14)
│   ├── predicted_vs_actual.py         # join_predicted_vs_actual() -- moved from api/routers/reporting.py
│   └── summary.py                      # build_report() -- FR-14.1/FR-14.2
├── api/
│   ├── routers/reporting.py         # Now a thin wrapper; +GET /reporting/summary, +GET /reporting/metrics
│   └── schemas.py                    # +ReportSummaryOut and 5 nested Out schemas
└── frontend/widgets/
    └── reporting_tab.py               # New tab (both modes) rendering /reporting/summary
```

**The refactor this phase did first: moving the join out of the router.**
Phase 10's `api/routers/reporting.py` docstring explicitly flagged this
as the intended next step once a second consumer needed the same join --
that second consumer is this phase's `build_prediction_accuracy_summary()`.
`reporting/predicted_vs_actual.py` is now the single source of truth for
the nearest-timestamp join (unchanged logic, just relocated + given a
`current_value` field the original didn't need); the router became a
thin wrapper matching the pattern every other subsystem's router already
follows (`routers/retraining.py` over `aiperfopt.retraining`, etc.). All
5 of Phase 10's existing `test_api_reporting.py` tests passed unchanged
against the refactored router before any new code was added, confirming
the move was behavior-preserving.

**Why `current_value` was added to the join, and where it's used.**
FR-14.2's prediction-accuracy summary needed directional accuracy
(rising/falling/stable relative to a reference point), and Section 11's
own definition of that metric --
`forecasting.evaluation.compute_directional_accuracy` -- takes a
`current` array as an explicit parameter. Rather than reimplement
directional-accuracy logic a second time for live-observed predictions,
the join was extended to also look up the actual value nearest each
prediction's `generated_at` (using the same `_nearest`/`bisect` helper,
over an actuals window now widened to cover both `generated_at` and
`target_timestamp` for every prediction in range) and
`summary.build_prediction_accuracy_summary` calls the *same*
`compute_mae_rmse`/`compute_directional_accuracy` functions Phase 4's
walk-forward validation harness uses. This means "prediction accuracy"
in a live report and "prediction accuracy" during offline evaluation
are defined identically, by construction, not just by convention.

**Why prediction-accuracy summaries are driven by what's actually in the
data, not a fixed metric list.** `build_system_metrics_summary` takes a
fixed default metric tuple (`DEFAULT_REPORT_METRICS`, Section 9's core
set) since system-metrics context should show up consistently report to
report. `build_prediction_accuracy_summary` instead queries which
`(metric_name, horizon_label)` combinations actually have a prediction
row in the window and reports exactly those -- a metric nobody trained a
model for, or that produced zero predictions this window, simply doesn't
appear as a row of `None`s. This was a deliberate asymmetry, not an
inconsistency: system metrics are collected unconditionally by
Monitoring, so a fixed list is meaningful context even in a quiet
window; predictions are not, so "nothing predicted" is better
represented by absence than by manufactured empty rows.

**ADL-22 compliance, made concrete.** The diagnosis summary zero-fills
every one of Section 12's eight real taxonomy categories
(`{c.value: 0 for c in DiagnosisCategory}`) before counting, so a report
consumer never has to special-case a missing key for a category that
simply didn't occur -- and, more to ADL-22's point, the report's shape
is fixed to exactly the three flagship pillars plus system-metrics
context; there is no generic "dump every table" endpoint anywhere in
this phase, matching the spec's explicit warning against reporting
becoming general-purpose dashboarding.

**FR-14.1's "exportable," resolved without inventing a new format.**
`?download=true` on `/reporting/summary` returns the identical JSON body
with a `Content-Disposition: attachment` header rather than adding a
CSV/PDF export path -- consistent with the previous handoff's own
suggestion (Section 8) to default to JSON-over-REST and treat a
downloadable file format as an explicit, separate scope decision rather
than something to silently add. If a future session wants CSV/PDF
export, that's a clean, additive follow-up on top of `build_report()`
(which already returns a fully-computed, framework-agnostic dataclass),
not a rework of this phase's aggregation logic.

**The frontend's `ReportingTab` is available in both modes, unlike
`RetrainingPanel`.** Deliberate: a report tying back to prediction
accuracy/diagnosis/recommendations is exactly the kind of "AI-generated
insight" Section 4 says General Mode should emphasize, whereas
Retraining's human-confirmed deployment machinery (Phase 8/9) is a
Developer Mode concern by nature of what it does, not just by
convention. The tab renders every report field with plain-sentence
formatting (`"3 diagnoses, 1 actionable | Avg confidence: 82%"`) rather
than the raw JSON a Developer Mode-only view might have gotten away
with.

**No config additions this phase.** `ForecastingSettings` has no
`target_metrics` field either (retraining passes it explicitly at call
time instead, per Phase 8), so the same precedent was followed here:
`DEFAULT_REPORT_METRICS` is a module-level constant in `summary.py`, and
`period`/`start`/`end` are request-time parameters, not new
`AppSettings` fields. Nothing about reporting cadence, retention, or
default metrics needed NFR-12.1's externally-configurable treatment
this phase, since nothing here is a long-running tunable the way
polling interval or contention threshold are -- it's a computation
triggered per-request.

---

## 4. Real bugs found and fixed during implementation

These are worth knowing about because they're exactly the kind of thing
that could get silently reintroduced if a future session "cleans up" the
code without this context:

1. **Pydantic private-attribute gotcha** (`config/settings.py`): a class
   attribute named `_user_config_path` gets intercepted by pydantic's
   metaclass as a `ModelPrivateAttr` descriptor, not a plain value —
   subclassing to override it silently doesn't work. Fixed with a
   `contextvars.ContextVar` instead. If you ever see underscore-prefixed
   class attributes added to a pydantic model for "passing data through,"
   that's this bug waiting to happen again.

2. **False launch events on startup** (`monitoring/process_tracker.py`):
   without a `_baseline_established` flag, the *first* poll would treat
   every already-running process on the machine as a "launch" event. Real
   launch detection only applies from the second poll onward.

3. **psutil CPU-percent measured against the wrong reference**
   (`monitoring/process_tracker.py`): `psutil.Process.cpu_percent()`
   compares against the *same Process object's* last call. Recreating
   `Process` objects every poll cycle (the naive approach) makes every
   reading meaningless. Fixed by caching `Process` objects across cycles
   in `ProcessTracker._processes` and priming new ones the cycle they're
   discovered (result discarded, real delta available next cycle).

4. **Pandas 3.0 datetime-resolution mismatch** (`common/timeutils.py`,
   affects both `features/engine.py` and `forecasting/dataset.py`):
   `pd.to_datetime(x, unit='s')` infers second-level resolution when
   every input happens to be a whole number, but `index - pd.Timedelta(...)`
   upcasts the result to microsecond resolution — the subsequent
   `pd.merge_asof` then raises `MergeError: incompatible merge keys`.
   Invisible with real `time.time()`-based timestamps (always has a
   fractional part) but reliably reproduces with synthetic/whole-second
   test data — which is exactly what Phase 4's scenario generator
   produces. This was a **latent bug in Phase 3's code**, only surfaced
   when Phase 4 started feeding it whole-second synthetic timestamps.
   Fixed with a shared `epoch_seconds_to_utc_datetime()` helper that
   pins resolution to nanoseconds unconditionally. If you ever see
   `pd.to_datetime(..., unit='s')` called directly anywhere instead of
   through this helper, that's this bug waiting to happen again.

5. **Walk-forward CV on a single synthetic scenario is not a fair
   evaluation** (`forecasting/training_pipeline.py`): found empirically
   while building this module. Training and testing on one deterministic
   scenario (e.g. just "cpu_stress," which ramps once and plateaus) means
   early chronological folds train only on the pre-ramp segment and test
   on the ramp/plateau they've never seen — making LightGBM look
   artificially far worse than the baselines. Fixed by combining multiple
   differently-shaped scenarios (computed with features independently
   per-scenario to avoid cross-scenario window leakage, then concatenated)
   before running `TimeSeriesSplit`. Documented in that module's
   docstring in detail since it's a genuine ML-correctness issue, not
   just a design preference.

6. **Counter-metric forecasting was predicting the wrong quantity
   entirely** (`forecasting/dataset.py`, `forecasting/training_pipeline.py`,
   `forecasting/baselines.py`): found while building Phase 5, not Phase 4
   — a real, retroactive bug in already-shipped code. For a counter
   metric (e.g. `disk_read_bytes`, a cumulative OS counter reaching into
   the millions), the label was set to the raw future counter value,
   while every feature available to the model was a small-magnitude rate
   (thousands, not millions) — the model had no feature carrying the
   current large baseline, so it was structurally unable to predict a
   similarly large future value. This also silently broke
   `MovingAverageForecaster` for every counter metric: it looked up a
   column name that never existed (`{metric}_roll_mean_*` instead of
   `{metric}_rate_roll_mean_*`), so it always silently fell through to
   its naive-persistence fallback without erroring. Fixed by adding
   `is_counter` throughout, which rate-transforms both the label and the
   baseline's column lookup. Caught only because Phase 5 needed real,
   usable counter-metric forecasts to feed Contention Prediction — Phase
   4's own tests never exercised a counter metric end-to-end. If you
   trust Phase 4's zip from before this fix, disk/net forecasts in it are
   unreliable; the fix is retroactive and included from Phase 5 onward.

7. **`ExponentialSmoothingForecaster` could silently return NaN**
   (`forecasting/baselines.py`): statsmodels' `Holt` does not reliably
   raise on NaN-contaminated input — it can silently propagate NaN
   through its computation and return a NaN forecast, which the original
   try/except never caught (NaN doesn't raise). Found training on
   counter-rate series that span concatenated synthetic scenarios (rate
   transform introduces NaN at scenario boundaries). A single NaN fold
   result then poisoned the *entire* aggregate for that model via
   `np.mean`'s NaN-propagation — fixed both at the source (explicit
   NaN-cleaning before fitting, explicit finite-value check on the
   output, in `baselines.py`) and defensively downstream
   (`mean_metrics_by_model` now uses `nanmean` with a warning log, so one
   bad fold can't silently zero out an otherwise-good aggregate again).

8. **Walk-forward CV on block-concatenated scenarios starves rare-event
   classifiers of positive examples** (`contention/training_pipeline.py`):
   found while evaluating Contention Prediction — precision/recall/F1
   came back all zero. Diagnosis: each synthetic scenario occupies one
   contiguous chronological block (large time offset between scenarios,
   per `synthetic.py`'s anti-leakage design), so a `TimeSeriesSplit` fold's
   test slice was often *one entire scenario in isolation* — a fold
   testing on `disk_io_burst` (which never crosses any contention
   threshold) has undefined recall by construction, and a fold trained
   only on memory-pressure examples has no way to recognize a
   qualitatively different CPU-spike pattern in its test set. This was
   invisible for Forecasting's continuous MAE (a block being "all one
   scenario" doesn't break a continuous metric the same way) but fatal
   for a rare binary label. Fixed with `_interleave_scenario_chunks()`:
   each scenario's rows are split into contiguous chunks and interleaved
   chunk-by-chunk across scenarios before CV, so every fold's train *and*
   test slice gets a representative mix of scenario shapes, without
   reordering any scenario's own internal chronology.

9. **Plain-language template overclaiming for CPU-Bound Single Process**
   (`diagnosis/text.py`, Phase 6): see the "Phase 6 addendum" section
   above (§3a) for the full writeup. Caught via manual smoke-testing
   before the formal Phase 6 test suite existed — worth noting as a
   reminder that a quick real end-to-end run before writing tests can
   catch things unit tests of individually-correct-looking pieces won't:
   `_process_names()` and the taxonomy resolver were each individually
   correct in isolation; the bug only appeared where they were combined
   in the CPU_SINGLE_PROCESS template specifically.

10. **`diagnose_contention_event()` discarded the persisted row's id**
    (`diagnosis/service.py`, caught while building Phase 7): `insert_diagnosis()`
    returns the int primary key SQLAlchemy assigned on insert, but the
    original code called it and threw the return value away
    (`insert_diagnosis(engine, record); return record`), so every
    `DiagnosisRecord` `diagnose_contention_event()` returned to a caller
    had `id=None` even though the row really was persisted with a real
    id. This wouldn't have caused a loud failure in Phase 6 alone — Phase
    6 never needed a diagnosis's own id — but it would have silently
    broken FR-9.3 the moment Phase 7 tried to write a `RecommendationRecord.diagnosis_id`
    from it. Caught the same way as bug #9: a manual end-to-end smoke
    test (real classifier, real forecaster, real storage engine) run
    *before* writing Phase 7's formal tests, specifically because Phase 7
    was the first consumer of `DiagnosisRecord.id` and therefore the
    first thing that could actually notice the bug. Fixed by capturing
    `insert_diagnosis()`'s return value and assigning it to `record.id`
    before returning. Regression test:
    `test_diagnosis_service.py::test_triggered_cpu_event_persists_single_process_diagnosis`'s
    `assert result.id is not None` line, plus the id round-trip already
    covered incidentally by every Phase 7 service test.

    **Worth flagging as a pattern, not just a one-off bug:** this is the
    second time in a row (bug #9, then this one) that a real integration
    bug surfaced specifically at the moment a *new downstream consumer*
    exercised a subsystem for the first time, rather than at any point
    while that subsystem's own tests were being written. Both `text.py`'s
    process-naming bug and this `.id` bug were individually invisible to
    Phase 6's own test suite because Phase 6 never needed the thing that
    was actually wrong (a single process's exclusivity in the sentence;
    the diagnosis's own database id). If Phase 8 (Retraining) or Phase 9
    (API) end up being the first real consumer of some other Phase 6/7
    return value nobody's exercised yet, running a real end-to-end smoke
    test before trusting that phase's own unit tests continues to be the
    right first move — cheaper than discovering it in front of an
    evaluator.

11. **Holdout evaluation rows were tagged with a synthetic-scenario-
    derived `dataset_tag`, unconditionally** (`forecasting/training_pipeline.py`,
    `contention/training_pipeline.py`; caught while building Phase 8).
    Both held-out generalization checks wrote
    `dataset_tag=f"holdout_{HELD_OUT_SCENARIO_NAME}"` (i.e.
    `"holdout_cpu_stress"`) unconditionally — harmless while the only
    data source was Phase 4/5's synthetic bootstrap, but actively
    misleading the moment Phase 8 started training against real-usage
    data and writing evaluation rows with the *same* hardcoded tag: a
    human (or `retraining/comparison.py`, which filters evaluations by
    `dataset_tag.startswith("holdout_")`) reading evaluation history
    later would see "holdout_cpu_stress" on a row that had nothing to do
    with the cpu_stress synthetic scenario. Caught before it could
    produce a wrong comparison, not after, because Phase 8's design
    explicitly reasoned through "what does `compare_candidate_to_active`
    actually need to find" before writing the retraining data source.
    Fixed by adding an optional `holdout_dataset_tag` parameter to both
    `train_evaluate_and_select_one`/`train_and_evaluate_all` and
    `train_contention_and_anomaly_pipeline`, defaulting to the exact
    previous hardcoded value when omitted (so every existing Phase 4/5
    test still passes unchanged) — `retraining/pipeline.py` passes
    `"holdout_real_usage"` explicitly for the real-usage path. Regression
    coverage: `test_retraining_pipeline.py`'s
    `test_run_retraining_on_real_usage_data_produces_candidate` and
    `test_run_retraining_on_synthetic_data_produces_candidate` both
    assert the exact tag written for their respective data source.

12. **Baseline forecasters need a `current_value` key on the row passed
    to `.predict()`, which the first `build_live_contention_input_row()`
    draft didn't supply** (`contention/dataset.py`; caught by an
    end-to-end smoke test before any formal test was written, Phase 9).
    The training path's per-metric row (`build_supervised_dataset`'s
    output) always carries that metric's own `current_value` alongside
    engineered features — `NaivePersistenceForecaster`/
    `MovingAverageForecaster.predict()` (`forecasting/baselines.py`)
    read it directly off the row rather than the engineered feature
    columns. The live-inference row builder initially passed the shared
    engineered `feature_row` straight to every metric's forecaster,
    which has no such column (it's specific to each metric, not a global
    feature) — any cycle where a baseline (not LightGBM) won a metric's
    forecast would have raised a `KeyError` on the very first live tick.
    Caught immediately by running the real trained pipeline end-to-end
    before writing `test_live_inference_plumbing.py`, not by the test
    itself — the smoke-test-first discipline earning its keep again (see
    bug #10's writeup). Fixed by copying `feature_row` per-metric inside
    the loop and setting `current_value` from that metric's own latest
    raw sample before calling `.predict()`. Regression coverage:
    `test_live_inference_plumbing.py`'s
    `test_build_live_contention_input_row_feeds_a_real_prediction` uses
    real trained forecasters (not just LightGBM) against a real
    prediction call, so a baseline-selected metric exercises this path.

13. **`LiveStateOut` couldn't actually be built from a `LiveState`
    dataclass** (`api/schemas.py`; caught by the full API smoke test,
    Phase 9). Every other response schema inherits from a shared
    `_RecordSchema` base with `model_config = ConfigDict(from_attributes=True)`;
    `LiveStateOut` was defined directly on `BaseModel` instead and never
    picked up that config, so `LiveStateOut.model_validate(state)`
    (used both for the WebSocket broadcast payload and the `/live/state`
    REST response) raised a validation error on the very first live
    cycle. Fixed by adding the same `model_config` directly to
    `LiveStateOut`. Regression coverage: every test in `test_api_live.py`
    round-trips a real `LiveState` through this exact conversion; the
    bug would fail all of them, not just one.

14. **WebSocket route dependencies can't reuse `Request`-typed HTTP
    dependency functions** (`api/dependencies.py`, `api/routers/live.py`;
    caught by the full API smoke test, Phase 9). `get_live_loop`/
    `get_connection_manager` were written to take `request: Request` —
    correct for ordinary HTTP routes, but FastAPI's WebSocket routes have
    no `Request` in their connection scope (only a `WebSocket`), so
    reusing those dependencies on `/ws/live` raised a dependency
    resolution error on connect. Not a subtle bug — every WebSocket
    connection would have failed immediately — but it only surfaces when
    something actually opens a WebSocket, which no unit test below the
    full-app level would exercise. Fixed by adding
    `get_live_loop_ws`/`get_connection_manager_ws` variants typed on
    `WebSocket` instead, used only by the WebSocket route. Regression
    coverage: `test_api_live.py`'s `test_websocket_receives_broadcast_state`
    and `test_websocket_connects_before_any_state_exists` both actually
    open a WebSocket connection through `TestClient`, not just call the
    dependency functions directly, so this class of bug can't hide behind
    a unit test that mocks the connection scope.

15. **`FeatureVectorOut`/`LiveStateOut.feature_vector` were typed as
    `dict[str, float]`, rejecting legitimate `None` values** (`api/schemas.py`;
    caught by the full API smoke test, Phase 9). Engineered features
    computed from a short rolling window (e.g. a `roll_std` over a
    lookback window with too few samples so far) are legitimately `NaN`/
    `None` — Phase 3's feature engine already handles this correctly
    (FR-4.3: "handle missing/gappy samples without producing silently
    corrupted features"), storing `None` rather than a fabricated number.
    The API schema didn't account for that: any feature vector containing
    even one `None` value (routine during the first `min_history_seconds`
    of a fresh live-inference run) failed Pydantic validation on the very
    first `/live/state` response. Fixed by widening both fields to
    `dict[str, float | None]`. Worth remembering if a future consumer
    (Phase 10's frontend) reads `feature_vector` directly: a `None` there
    is expected, ordinary behavior early in a session, not a data quality
    problem to report to the user.

16. **`ApiClient`'s shared 5-second request timeout was too short for
    `/retraining/run`** (`frontend/api_client.py`; caught by
    `tests/frontend/test_retraining_panel.py`'s real end-to-end
    round-trip test, Phase 10). The shared `httpx.Client` timeout is
    sized for ordinary fast local reads (matching NFR-3.3's <=500ms
    target); a real retraining call is a legitimately multi-second-to-
    multi-minute request even against the deliberately-shrunk test
    settings (PROJECT_HANDOFF.md Section 6's fast-settings pattern), and
    the shared default fired a timeout well before training finished,
    surfacing as a silent-looking failure in `RetrainingPanel` (the
    request is already off the UI thread via `workers.run_in_background`,
    so nothing froze — it just failed with a timeout that looked like a
    backend error). Fixed by giving `run_retraining()` its own
    request-level `timeout=600.0` override rather than raising the
    shared client default for every call, since every *other* endpoint
    genuinely is fast and a much longer shared default would just mask
    a real slow-response regression elsewhere.

---

## 5. Design decisions flagged for your awareness (not blocking, but you should know)

- **Process-level data is pruned, not rolled up**, past the retention
  window (`storage/rollup.py`). System-level metrics *are* rolled up into
  hourly aggregates. Rationale: per-process rollup requires tracking
  per-pid identity across buckets, real complexity for data whose value
  is concentrated in the recent window. Flag if you'd rather it rolled up
  too.
- **`raw_system_metrics.thread_count_total` is always `None`.** No
  lightweight system-wide psutil primitive provides it without a full
  process enumeration (which runs on a coarser interval). Section 9's
  "thread counts" requirement is satisfied via per-process `num_threads`
  in `raw_process_metrics` instead.
- **The `app_logs` storage table is *not* wired to a `logging.Handler`.**
  It exists (FR-3.1 requires it) but is meant for subsystems to
  explicitly persist *significant* events (e.g. a retraining outcome, a
  high-confidence contention prediction), not to mirror every file log
  line — doing that would violate the <2-3% CPU overhead budget (NFR-1.1).
  Nothing writes to this table yet; the first subsystem that wants
  queryable event history (likely Retraining or Diagnosis) should be the
  first real caller.
- **Feature Engineering's rate-of-change formula uses the window's
  *nominal* length as the time base**, not the lagged sample's actual
  offset (which can differ by up to `tolerance_seconds`). Documented as a
  deliberate simplification in `features/engine.py`.
- **Forecast horizons are trained against one representative point per
  range**, not literally every second in [30,60] and [120,300]
  (`config/settings.py`'s `short_horizon_target_seconds`/
  `medium_horizon_target_seconds` — the midpoint of each configured
  range: 45s and 210s by default). ADL-06 calls for "a single multi-step
  model per metric" but doesn't fully specify what "multi-step" means at
  the range level; midpoint-as-target is a defensible reading, not a
  spec-mandated one — flag if you'd rather see genuine multi-step output
  (e.g. a model predicting a trajectory of points across the range).
- **Confidence scores are one scalar per trained model** (derived from
  that model's walk-forward validation RMSE), not predicted per-input via
  quantile regression. This is deliberately the simpler of two reasonable
  designs — see `forecasting/model.py`'s docstring for why it matches
  ADL-11 and the spec's own V1.5 roadmap (calibration is explicitly a
  *future* enhancement on top of a simpler V1 baseline).
- **On the current synthetic training data, LightGBM does not always
  win.** In actual runs, `moving_average` or `exponential_smoothing`
  frequently outperform LightGBM's MAE on these modest synthetic
  datasets, and the pipeline correctly falls back to them per Section
  10.7's contingency plan. This is a real, honest evaluation outcome, not
  a bug — the harness is doing exactly what ADL-35 asks. Whether
  LightGBM would win with more/richer synthetic data (or real usage data
  in Phase 2/3 of Section 10.3) is an open question, not yet answered.
- **`synthetic.py` was promoted** from `forecasting/synthetic.py` to a
  top-level module in Phase 5, exactly as flagged as a possibility in
  this document after Phase 4 — Contention Prediction and Anomaly
  Detection turned out to be the "second consumer" that justified it.
- **Contention Prediction's walk-forward precision/recall is genuinely
  weak** (roughly 0.05-0.5 precision/recall depending on the run, not a
  clean number) even after the block-concatenation fix (bug #8 above).
  Root cause: the current default scenario set produces very few
  contention-positive examples (dozens out of ~1700 rows) concentrated in
  only two scenario shapes (`memory_pressure_gradual`,
  `benign_spike`'s CPU ramp) — a genuinely hard, severely
  class-imbalanced learning problem for a linear classifier with this
  little data. This is disclosed transparently rather than hidden,
  consistent with how Phase 4's baseline-beats-LightGBM finding was
  handled — but unlike that finding, this one likely *does* warrant
  further work (richer/more varied contention scenarios, more training
  data, or accepting real-usage data per Section 10.3 Phase 2 before this
  classifier is trustworthy for a live demo). Flagging this explicitly
  as the one place in Phase 5 where "architecturally correct and honestly
  evaluated" doesn't yet mean "good enough to demo confidently."
- **Contention labeling thresholds were tuned empirically** against the
  actual synthetic scenario data (`contention/config.py`'s
  `ContentionLabelingConfig` defaults: cpu_percent 70, mem_percent 85 —
  lowered from an initial untested guess of 80/90 specifically because
  the higher thresholds produced almost no positive training examples at
  all). If synthetic scenarios change in a future session, these
  defaults should be re-validated the same way (check what fraction of
  samples actually cross each threshold before trusting the default).
- **Contention Prediction only covers the short horizon** (not medium),
  matching the flagship demo's "within the next 30-60 seconds" framing.
  Medium-horizon contention prediction is a natural, low-risk extension
  (the plumbing already supports it — just pass medium-horizon outcomes
  instead) but wasn't built now to keep Phase 5's scope bounded.
- **`get_contributing_processes` falls back to CPU for counter-derived
  dominant metrics** (`contention/processes.py`) — Section 9's
  process-level metric set has no per-process disk/network I/O column in
  the current schema, so "which processes are driving high disk I/O" is
  answered with "which processes are busiest overall" rather than a
  precise per-process I/O attribution. A real limitation, not a bug;
  extending `raw_process_metrics` with per-process I/O counters (if
  psutil exposes them reliably) would be the fix, deferred as out of
  scope for Phase 5.
- **Anomaly Detection has a mild "cold start" artifact**: the very first
  row of any scenario has no lag/rolling history yet, so its NaN-filled
  features look different from the rest of the training distribution and
  can score as artificially anomalous. Noted, not fixed — low practical
  impact (one row per scenario) but worth knowing if a demo's very first
  live reading shows a surprisingly high anomaly score.
- **The synthetic scenario library doesn't produce a training/demo
  example for every Section 12 taxonomy category** (`synthetic.py`'s
  `idle`/`cpu_stress`/`memory_pressure_gradual`/`disk_io_burst`/
  `benign_spike` set has no network-saturation or sudden-memory-spike
  shape, and no scenario deliberately drives *several* processes to high
  CPU at once rather than one). Phase 6's taxonomy resolver is unit-
  tested directly against hand-constructed `ContentionPredictionRecord`
  inputs for every category (`test_diagnosis_taxonomy.py` covers all 8),
  so the *logic* is verified — but there is currently no live-demo path
  that would actually produce a real Network Saturation or Memory
  Pressure — Sudden diagnosis end-to-end from a running scenario. Adding
  scenario shapes for the missing categories (or accepting real-usage
  data, Section 10.3 Phase 2) would close this gap; flagged rather than
  fixed now to keep Phase 6 scoped to Diagnosis itself.
- **The active-model-version pointer is a plain JSON file, not a
  database row** (`retraining/active_version.py`). Deliberate: the live
  inference path (Phase 9) needs to answer "what's active right now" on
  every cycle, and a file read is a simpler and cheaper way to answer a
  single fact than a query. The durable, queryable *history* of every
  deployment decision still lives in the `model_deployments` table —
  the file is only ever a fast cache of what that history currently
  says. Flag if a future session would rather this be a table row for
  transactional consistency with other storage writes; the file-based
  approach trades that off for simplicity and read speed on the hot
  path.
- **`RetrainingComparison.recommendation` has three states, not two**
  (`"deploy"` / `"reject"` / `"no_active_baseline"`), and
  `"no_active_baseline"` deliberately does *not* mean "safe to
  auto-deploy" — it means there's nothing to compare against yet
  (typically the very first retraining run). `confirm_deployment()`
  still requires the same explicit `confirmed=True` regardless of which
  recommendation preceded it; the three-way split exists purely so a
  human reviewing the comparison can tell "this looks like a genuine
  improvement" apart from "there was nothing to measure this against."
- **Regression detection compares candidate holdout metrics against the
  active version's own *stored* holdout evaluation row**, not against a
  freshly re-run evaluation of the active model on the candidate's
  holdout data. This is simpler and faster (no need to reload and re-run
  the currently-deployed model), and is a reasonable comparison since
  both the candidate and the active version's stored result came from
  each one's own held-out generalization check (Section 11) — but it
  does mean the two holdout sets are not necessarily identical data
  (the active version's holdout was whatever data it was trained
  against at the time; the candidate's holdout is the current
  retraining run's chronological tail). Worth knowing if a comparison
  ever looks surprising: it's comparing "how well did each model
  generalize to its own held-out slice," not "how do both models
  perform on the exact same fixed test set."
- **No API/scheduling trigger for retraining exists yet** — Phase 8
  provides the callable pipeline (`run_retraining_on_real_usage_data`)
  and the confirmation/rollback primitives, but nothing currently calls
  them automatically or exposes them over HTTP. FR-11.1 explicitly
  allows "manually or scripted-triggered" — a cron job calling the
  Phase 8 entrypoints directly would already satisfy the requirement as
  written. Phase 9 (API Layer) is the natural place to add an actual
  "trigger retraining" / "confirm deployment" endpoint pair, matching
  how Diagnosis and Recommendation already take `model_version` as an
  explicit parameter rather than inferring it internally.
  **(Resolved in Phase 9 — see below; left here as the historical record
  of what Phase 8 deferred and why.)**
- **The live-inference loop is threading-based, and its only
  synchronization primitive is a plain lock around the latest snapshot**
  (`api/live_state.py`). No queue, no backpressure handling beyond "the
  next tick overwrites the previous snapshot" — deliberately, since only
  the *latest* state is ever meaningful for this use case (a stale
  half-second-old prediction has no value once a newer one exists,
  unlike Monitoring's queue-based design where every sample must be
  preserved for storage). If a future phase needs a full history of
  every live-inference tick rather than just the latest, that's a
  different data flow (probably: persist every cycle's summary to
  storage, not just hold the latest in memory) — not a change to this
  loop's synchronization.
- **The live-inference loop persists its own `ContentionPredictionRecord`
  to storage (`insert_contention_prediction`), but not a per-cycle "no
  prediction, nothing deployed yet" record.** Only cycles that actually
  reach contention prediction leave a storage trail; a gap in
  `/contention`'s history could mean either "nothing was running" or
  "something was running but every cycle degraded before reaching
  prediction" — those aren't currently distinguishable from stored data
  alone (only from `LiveState.error`, which isn't persisted). Worth
  addressing if the Reporting phase (Section 7.1 item 14) ever needs to
  reconstruct "was the live pipeline actually healthy over this time
  range," since right now that requires reading application logs, not
  just the `contention_predictions` table.
- **`/retraining/run` runs training synchronously, blocking the HTTP
  request until it completes** (which can be tens of seconds to a few
  minutes for a real training window, per Section 14's own performance
  notes). Acceptable for V1's course-timeline scope and consistent with
  every other Phase 8 entrypoint being a direct blocking call — but a
  future version wanting this triggered from a UI button without a
  multi-minute hung request would want a background-job pattern (return
  immediately with a job id, poll or WebSocket-push the result) rather
  than extending this endpoint's current synchronous shape.

---

## 6. Testing/quality notes

- Monitoring Engine and Ingestion Consumer tests use **real background
  threads** with short (0.05–0.1s) intervals, not mocked timers — this
  found real bugs (the two above) that mocks would have hidden. Keep this
  pattern for Phase 4+ if any new subsystem runs its own thread/loop.
- Process launch/termination detection is tested against **real spawned
  subprocesses** (`subprocess.Popen(["sleep", ...])`), not synthetic pid
  lists, for the same reason.
- Feature Engineering's FR-4.2 (no train/serve skew) is tested by
  **directly comparing** the batch-computed matrix's last row against the
  live-path's single-vector output on the same underlying data — not just
  asserted by code structure.
- Forecasting's walk-forward harness and full training pipeline are
  tested with **real LightGBM training and real statsmodels fitting**,
  not mocked — deliberately scaled down (short scenario durations, fewer
  walk-forward splits, small `min_training_rows`) to keep the suite fast
  (~15s for the full-pipeline tests) rather than mocking away the actual
  numerical behavior, which is exactly what caught bug #5 above.
- **Exponential Smoothing (Holt's method) is the slow part** of any
  evaluation involving it — it fits a fresh statsmodels model per test
  row. A full production training run across all 8 metrics × 2 horizons
  with the default 900s scenario duration takes on the order of a few
  minutes, not seconds. Fine for an explicit training/retraining
  invocation; not something to call on a tight request/response path.
- **Anomaly Detection, Contention Prediction, and their training pipeline
  are also tested with real fitting** (real `IsolationForest`, real
  `LogisticRegression`, real `Holt` where ETS is the deployed forecaster)
  — this is what surfaced bugs #6, #7, and #8 above. The full
  Forecasting→Contention integration test
  (`test_contention_training_pipeline.py`) takes ~46s by itself; the
  complete 258-test suite runs in ~80s total.
- **Phase 6's SHAP-related tests use real `shap.TreeExplainer` and real
  `shap.LinearExplainer` calls**, not mocks — including the equivalence
  test proving contention attribution matches real SHAP output to 1e-9
  (see §3a above). `test_diagnosis_service.py` exercises the full
  Forecasting-record → Contention-record → Diagnosis path with a real
  trained `MetricForecaster` and `ContentionClassifier`, not stubs.
- Before writing the formal Phase 6 test suite, a manual end-to-end
  smoke test (real classifier, real forecaster, real storage engine) was
  run first specifically to catch integration issues unit tests of
  individually-correct pieces wouldn't — this is what caught bug #9.
  Worth repeating as a step for any future subsystem that reconciles
  several upstream subsystems' output the way Diagnosis does.
- **Phase 7's catalog tests validate the real packaged
  `config/recommendations.yaml`** (not a synthetic fixture standing in
  for it) against every actionable `DiagnosisCategory` — if a future edit
  to the YAML drops a category or breaks an action's structure, the test
  suite fails on the actual file that ships, not on a copy that could
  silently drift from it. `test_recommendation_service.py`'s
  `test_every_actionable_category_round_trips_through_the_real_catalog`
  does the same for the service layer: every category from the real
  catalog produces a real persisted recommendation, not just the one or
  two categories a hand-picked example might have covered.
- Same "smoke test before formal tests" discipline was applied again
  building Phase 7, and it's what caught bug #10 — see bug #10's writeup
  above for why this is turning into a pattern worth keeping, not a
  one-off precaution.
- **Phase 8's integration tests exercise the real Phase 4/5 training
  pipelines end-to-end against hand-constructed real-usage-shaped data**
  (`test_retraining_pipeline.py`) — real LightGBM training, real
  contention classifier fitting, a real confirmed deployment, and a real
  second retraining run whose comparison looks up the first run's
  actually-persisted holdout evaluation rows from storage. Deliberately
  *not* built on the `aiperfopt.synthetic` scenario generators — that
  module's own docstring documents synthetic samples as never belonging
  in `raw_system_metrics` (mixing them in would corrupt exactly the real
  data Phase 8 is supposed to retrain on), so the tests construct a
  simple rising-CPU-with-noise series directly instead, matching the
  precedent already set by `test_retraining_real_usage_data.py`.
- `test_retraining_comparison.py` and `test_retraining_deployment.py`
  test the comparison/confirmation logic in isolation from real ML
  fitting (hand-built `TrainingRunSummary`/`ContentionTrainingSummary`
  objects and directly-inserted `EvaluationRecord` rows) — deliberately,
  since that logic's correctness doesn't depend on what specific numbers
  a real LightGBM run produces, only on how it reacts to given numbers.
  The one full end-to-end regression-detection round trip (`train once,
  confirm, retrain again, compare against what got confirmed`) is
  covered by `test_retraining_pipeline.py`'s
  `test_second_candidate_compares_against_first_deployed_version`, so
  the wiring between the two layers is still verified for real, just not
  duplicated across every comparison-logic edge case.
- **Phase 9's API tests use real wall-clock timestamps for any test that
  exercises the live-inference loop** (`tests/api_conftest.py`'s
  `rising_cpu_samples_now()`), unlike every earlier phase's synthetic
  fixtures (which use arbitrary/relative timestamps starting at 0 — fine
  for training/evaluation code, which always takes an explicit `as_of`).
  The live loop calls `get_recent_system_metrics`/
  `compute_and_store_live_feature_vector` without an explicit `as_of`,
  reading relative to genuine `time.time()` — this was actually the
  first thing the Phase 9 smoke test caught (before any of the four bugs
  in the registry above): synthetic epoch-0 timestamps produced an empty
  "recent" window against real wall-clock "now," so nothing downstream
  ever ran. Worth remembering for Phase 10: any test/demo data meant to
  exercise the *running* system (as opposed to a one-shot training/batch
  call) needs to be anchored to real time, not an arbitrary clock.
- **`tests/api_conftest.py` is deliberately not named `conftest.py`** —
  its fast-settings/app-building helpers are specific to the API test
  files (`test_api_*.py`) and imported explicitly by each one, rather
  than being auto-discovered as fixtures for the entire suite. The
  existing `tests/conftest.py` (used by every other test file) is
  untouched.
- **Phase 10 extends the same "real, not mocked" discipline to the
  frontend, adapted for what a Qt client actually needs.**
  `tests/frontend/server_fixture.py` starts a real `uvicorn.Server` on a
  real local port for every test that needs `ApiClient`/
  `LiveWebSocketClient` to talk to something — including a genuine
  reconnect test (`test_reconnects_after_backend_restarts` in
  `test_live_client.py` points the client at a port nothing is
  listening on and confirms it keeps retrying without crashing, rather
  than asserting reconnect logic exists by code inspection) and a real
  end-to-end retraining round-trip through the actual widget
  (`test_retraining_panel.py::test_run_and_confirm_full_round_trip` runs
  real LightGBM training via the real `/retraining/run` endpoint, not a
  stub response). Widget-only tests that don't need a backend
  (`test_dashboard_widget.py`, `test_developer_panel_widget.py`,
  `test_predicted_vs_actual_widget.py`) construct widgets directly and
  feed them hand-built `LiveStateOut`-shaped dicts instead, since the
  question there is "does this widget render this payload correctly,"
  not "does the network path work" — matching the same instinct as
  Phase 9's split between comparison-logic-only tests and full pipeline
  tests (Section 6, `test_retraining_comparison.py` vs.
  `test_retraining_pipeline.py`, one paragraph up).
- **`pytest-qt`'s `qtbot` fixture plus `QT_QPA_PLATFORM=offscreen` runs
  the entire Qt suite headless** — no real display server needed,
  including for `QtCharts`/`QWebSocket` widgets, which was confirmed to
  work in this sandboxed dev environment before any widget code was
  written (worth re-confirming in whatever environment picks this
  project up next; some minimal CI containers omit the Qt platform
  plugins `offscreen` mode needs).
- **Two tests are flaky specifically under heavy concurrent CPU load,
  confirmed environment contention rather than a real bug**: a full
  366-test suite run on a resource-constrained machine occasionally
  flakes `tests/frontend/test_retraining_panel.py::test_run_and_confirm_full_round_trip`
  (a `qtbot.waitUntil` timeout waiting for a background REST call to
  land) and, separately, the pre-existing (Phase 2, untouched by this
  phase) `tests/test_monitoring_process_tracker.py::test_real_subprocess_termination_is_detected`
  (a spawned subprocess's reported name occasionally doesn't match under
  contention). Both pass reliably run alone or under normal load; the
  second one flaking too, despite being nothing Phase 10 touched,
  confirms this is a property of running many CPU-heavy tests (real
  LightGBM/statsmodels fits, real subprocess spawns, real background
  threads) back-to-back on a constrained machine, not something specific
  to the new frontend code. Worth re-running the specific failing test
  alone before assuming a real regression if this recurs.
- **Phase 11's full-suite run (402 tests) showed neither of the two
  flaky tests above** -- consistent with them being environment
  contention rather than a real regression, though this doesn't prove
  they can't recur; the guidance above (re-run the specific failing test
  alone before assuming a real regression) still applies.
- **`test_reporting_summary.py` and `test_reporting_predicted_vs_actual.py`
  are pure-subsystem tests with zero API/FastAPI involvement** -- they
  call `aiperfopt.reporting.summary`/`predicted_vs_actual` functions
  directly against the shared `storage_engine` fixture, the same split
  Phase 8 established between `test_retraining_comparison.py` (pure
  logic) and `test_retraining_pipeline.py` (full orchestration). The
  existing `test_api_reporting.py` then only needs to confirm the API
  layer wires the subsystem correctly (status codes, query-param
  parsing, response shape) rather than re-proving the join/aggregation
  math a second time through HTTP.
- **`test_prediction_accuracy_summary_computes_mae_and_directional`
  (in `test_reporting_summary.py`) is worth reading if extending
  accuracy reporting later** -- it's the one test that exercises the
  full `current_value` -> `compute_directional_accuracy` path end to
  end with a hand-constructed scenario where both predictions correctly
  call a rise, asserting `directional_accuracy == 1.0` exactly, which is
  a stronger check than an approximate/range assertion would give and
  was worth the extra care given this metric reuses Section 11's own
  definition rather than a new one.

---

## 7. Workflow preferences established in this conversation

- Zip files are provided **only when explicitly requested** ("give me the
  zip"), not automatically after every milestone.
- Development follows the phased roadmap in `README.md` / `PROJECT_SPEC.md`
  Section 20, strictly in dependency order — no phase starts before its
  dependencies are confirmed complete.
- Each milestone gets a review against spec compliance before moving on
  (see the "Milestone review" sections in prior chat responses — not
  reproduced here, but the pattern: check FRs/NFRs/ADLs touched, flag any
  scope decisions made, state what's next).
- This document (`PROJECT_HANDOFF.md`) is updated at the end of every
  milestone, not just when explicitly asked — that update is now a
  standard part of finishing a phase, alongside the code itself.

---

## 8. Next milestone

**All 11 Section 7.1 subsystems are now at least minimally implemented
-- V1 is functionally complete against the subsystem map.** There is no
mandatory "Phase 12" the way each previous phase had a clear next
subsystem waiting. What's left is either (a) a formal spec-compliance
pass, or (b) picking items from `PROJECT_SPEC.md` Section 20's post-V1
roadmap. Recommended order:

- **Do the spec-compliance pass first, before adding anything new.**
  Two previous handoffs (Section 8 of the Phase 9 and Phase 10 writeups)
  both flagged this and both deferred it in favor of finishing the
  remaining subsystem. Now that none remain, go through every FR/NFR/ADL
  in `PROJECT_SPEC.md` section by section and confirm each is actually
  satisfied by the shipped code, not just plausibly satisfied by
  memory -- this project's own governing principle (Section 3:
  "correctness over novelty," "if a simpler model performs comparably...
  the decision documented transparently") argues for verifying rather
  than assuming at this milestone. A few areas worth specific attention
  given how the phases actually built out:
  - **NFR-1.1/1.2 (resource overhead: <2-3% CPU, ~200-300MB memory)**
    has not been explicitly measured against a running instance anywhere
    in this handoff's history -- worth an actual measurement pass with
    Monitoring + live inference + the frontend all running together.
  - **NFR-3.1 (end-to-end pipeline latency <=1s)** likewise hasn't been
    explicitly timed end-to-end (sample -> updated prediction available)
    against a live instance, as opposed to individual subsystem unit
    tests.
  - **Section 15's testing-strategy categories** (User Acceptance
    Testing in particular: "does a non-technical observer understand the
    health/insight/recommendation narrative") are inherently not
    something `pytest` verifies -- this would need an actual live demo
    run-through, ideally by someone other than whoever wrote the code.
- **If instead continuing to build (post-V1, Section 20's "Version
  1.5"/"Version 2" roadmap), the lowest-risk, highest-value next items
  given what's already in place are:**
  - **Statistical calibration of confidence scores** (reliability
    diagrams, Platt/isotonic scaling) -- V1 deliberately deferred this
    (ADL-11) in favor of conservative thresholds; the walk-forward
    evaluation harness (Phase 4) and its stored per-fold results already
    provide the raw material a calibration pass would need.
  - **The `aiperfopt/scenarios/` Scenario Library pipeline** flagged at
    the end of Phase 5 (see memory: "proposed to address data generation
    weaknesses") -- if contention prediction precision/recall is still a
    concern by this point, this is the documented fix, not a new idea.
  - **CSV/PDF export for `/reporting/summary`** -- explicitly deferred
    this phase (Section 3f) as an additive follow-up on top of
    `build_report()`'s already-framework-agnostic dataclass, not a
    rework.
  - **Wiring the frontend's initial mode toggle to the backend's
    configured `mode.active_mode`** -- flagged as a reasonable small
    follow-up in Section 3e, still not done.
- **Whichever direction is chosen, update this document's Section 2
  status table and this "Next milestone" section accordingly** -- the
  table's current "11/11 complete" state is itself worth preserving
  verbatim in the next handoff's summary paragraph, since it's a
  meaningful project-level milestone (V1 functional completeness), not
  just routine phase progress.
