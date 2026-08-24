# AI System Performance Optimizer

Predictive, explainable macOS system health monitor. Full specification:
`PROJECT_SPEC.md` (single source of truth — see repository root / project
files).

## Status

**Phase 0 — Project Foundation: complete.**
**Phase 1 — Data Storage: complete.**
**Phase 2 — Monitoring Engine + Data Ingestion: complete.**
**Phase 3 — Feature Engineering: complete.**
**Phase 4 — Forecasting Models + Evaluation Harness: complete.**
**Phase 5 — Contention Prediction + Anomaly Detection: complete.**
**Phase 6 — Diagnosis / Explainable AI: complete.**
**Phase 7 — Recommendation Engine: complete.**
**Phase 8 — Model Retraining: complete.**
**Phase 9 — API Layer: complete.**
**Phase 10 — Frontend: complete.**
**Phase 11 — Reporting / Analytics: complete.**

Implemented so far:
- Layered configuration (`aiperfopt.config`) — packaged defaults → user
  override file → `.env` → environment variables (FR-15.1).
- Structured JSON logging (`aiperfopt.applog`) — one JSON object per line,
  console + rotating file handler, logger name doubles as pipeline-stage
  tag (FR-16.1).
- Storage engine bootstrap (`aiperfopt.storage.engine`) — SQLAlchemy Core
  engine over SQLite with WAL journaling and foreign keys enabled
  (ADL-17).
- Full storage schema (`aiperfopt.storage.schema`) — `raw_system_metrics`,
  `raw_process_metrics`, `process_events`, `raw_system_metrics_rollup`,
  `features`, `forecast_predictions`, `contention_predictions`,
  `evaluations`, `app_logs` (FR-3.1), every timestamp column indexed
  (FR-3.3).
- Typed read/write API (`aiperfopt.storage.reader` /
  `aiperfopt.storage.writer`) — the only supported way to touch storage;
  no other subsystem should build raw SQL against `schema.py` directly.
- Downsampling + retention (`aiperfopt.storage.rollup`) — old system
  metrics are aggregated into hourly buckets and pruned in one
  transaction; old process-level data is pruned outright (FR-3.2,
  NFR-4.1, NFR-4.3). Wired to a periodic check inside `IngestionConsumer`
  (Phase 2).
- **Monitoring Engine** (`aiperfopt.monitoring`) — two dedicated polling
  threads (system metrics, process metrics) producing onto a shared
  bounded queue (FR-1.1, FR-1.2, ADL-18). Per-metric-source graceful
  degradation (FR-1.4) and real process launch/termination detection via
  a persistent pid→Process cache (FR-1.3), verified against real
  subprocesses in tests. Process exclusions (Section 13, NFR-7.4) are
  enforced at collection time — excluded processes never reach the queue.
  Backpressure is handled by dropping samples (never blocking) if the
  queue fills (FR-2.2).
- **Data Ingestion** (`aiperfopt.ingestion`) — a single consumer thread
  drains the shared queue in strict FIFO order, batches same-type records,
  and flushes to storage on batch-size or time triggers (FR-2.1, FR-2.3).
  Storage failures during flush are logged and the batch is dropped
  rather than crashing the consumer or blocking Monitoring (NFR-5.4).
  Also owns the periodic retention-policy trigger.
- **Feature Engineering** (`aiperfopt.features`) — lag features, rolling
  statistics, and rate-of-change over the Section 9 metric set (FR-4.1).
  One pandas-vectorized function (`features.engine.compute_feature_matrix`)
  is the single source of truth used by both the live-inference path and
  batch/training path, structurally guaranteeing no train/serve skew
  (FR-4.2) — verified directly in tests, not just by code review. Gap
  handling (FR-4.3): lag lookups use a time tolerance and return `None`
  rather than a stale value across a gap; OS counters (disk/net I/O) are
  rate-transformed before any statistics are computed on them, with
  counter resets treated as gaps.
- `aiperfopt.pipeline.create_monitoring_pipeline()` — assembles Monitoring
  + Ingestion around their shared queue; deliberately *not* auto-started
  by `bootstrap()`, so importing the package never has the side effect of
  spawning threads.
- **Forecasting Models + Evaluation Harness** (`aiperfopt.forecasting`) —
  LightGBM per metric/horizon (ADL-13) with model-derived confidence
  from walk-forward validation RMSE (FR-5.2, ADL-11); three mandatory
  baselines (naive persistence, moving average, Holt exponential
  smoothing — Section 10.8) sharing one `Forecaster` interface with the
  real model (FR-5.3); walk-forward validation via `TimeSeriesSplit`
  (Section 11, FR-10.1) with MAE/RMSE/directional accuracy; Section
  10.7's contingency plan (fall back to the best baseline if LightGBM
  doesn't meaningfully outperform it) implemented and evaluation-result
  driven, not hardcoded. A synthetic scenario generator (`aiperfopt.synthetic`,
  Section 10.3 Phase 1) supplies training/eval data since no real usage
  history exists yet; one scenario is held out entirely from training
  folds for a genuine generalization check (Section 11). Evaluation
  results persist to the `evaluations` table; trained LightGBM artifacts
  persist to disk under `AppSettings.forecasting.models_dir`.
- **Contention Prediction + Anomaly Detection** (`aiperfopt.contention`,
  `aiperfopt.anomaly`) — Isolation Forest (Section 10.4, FR-7.1) scores
  the same engineered feature vectors Forecasting consumes; a logistic
  regression classifier (Section 10.4, FR-6.1) consumes *only* forecast
  outputs + current state + the anomaly score — never raw metrics
  directly — and its coefficients give free per-feature attribution for
  FR-6.2's "contributing metrics," with contributing *processes* resolved
  via a lookup against `raw_process_metrics` once the dominant metric is
  known. Anomaly score dampens the reported contention probability
  (FR-7.2) and significant anomalies emit a distinct notification into
  `app_logs`, never routed through the Recommendation Engine (ADL-36).
  Walk-forward precision/recall/F1 (Section 11's methodology, adapted for
  classification) persist into `evaluations.extra`.
- **Diagnosis / Explainable AI** (`aiperfopt.diagnosis`) — reconciles
  Contention Prediction and Forecasting output into exactly one of
  Section 12's eight taxonomy categories (FR-8.3, ADL-24): a rule-based
  resolver (`taxonomy.py`) checks for a significant anomaly first
  (ADL-36 — that category always wins and bypasses the Recommendation
  Engine), then finds the dominant resource type among contributing
  metrics, splitting CPU-dominant events into single- vs. multi-process
  and memory-dominant events into gradual vs. sudden. Explanation
  generation is genuinely tiered (Section 10.5, NFR-1.3) for Forecasting
  attribution — native LightGBM feature importance by default, real
  `shap.TreeExplainer` only on high-confidence contention predictions,
  significant anomalies, or explicit Developer Mode request — with an
  honest fallback note when Section 10.7's contingency plan means the
  deployed forecaster is a baseline, not LightGBM, and therefore has no
  per-feature attribution to give. Contention attribution is reused as-is
  rather than re-computed through SHAP: `ContentionClassifier`'s existing
  coefficient-based attribution is numerically verified (in tests) to be
  the exact SHAP-linear decomposition already. Every diagnosis gets both
  a technical (Developer Mode, FR-8.1) and plain-language (General Mode,
  FR-8.2) explanation from the same underlying data, the latter including
  a short sentence connecting the symptom to the OS mechanism behind it
  (Section 6's "OS concept demonstration" requirement). Persists to the
  `diagnoses` table.
- **Recommendation Engine** (`aiperfopt.recommendation`) — a deliberately
  thin, rule-based subsystem (ADL-15): a pure `category -> ranked actions`
  lookup against `config/recommendations.yaml` (NFR-12.1's "externally
  configurable," not a Python dict literal), gated strictly on
  `DiagnosisRecord.is_actionable` (`False` only for the anomaly category,
  ADL-36 — checked directly rather than re-derived from the category
  string). No ML, no thresholds, no raw metrics consulted (FR-9.1); no
  system-mutating action is ever taken, only data is produced and
  persisted (FR-9.2). Every recommendation carries the originating
  diagnosis's id and category for traceability (FR-9.3). The catalog
  loader validates at load time that every actionable Section 12 category
  is covered and that the anomaly category is never present, so a
  misconfigured catalog fails loudly at startup rather than surfacing as
  a confusing runtime `None`. Persists to the `recommendations` table.
- Shared exception hierarchy (`aiperfopt.common.exceptions`) used by every
  subsystem for consistent, gradeable error handling.
- **Model Retraining** (`aiperfopt.retraining`) — reuses Phase 4/5's
  training pipelines unchanged (FR-11.1) against accumulated real-usage
  history (`raw_system_metrics`, Section 10.3 Phase 2) instead of a fresh
  synthetic bootstrap: a chronological (never leaking future-into-past,
  ADL-10) train/holdout split of that history is built and handed to the
  existing `train_and_evaluate_all`/`train_contention_and_anomaly_pipeline`
  entrypoints via their new `scenarios`/`holdout_samples` parameters. The
  resulting candidate's holdout metrics are compared against whatever
  model_version is currently active (its own previously-recorded holdout
  evaluation, looked up from the `evaluations` table) — a regression on
  any comparable metric beyond a configurable tolerance recommends
  against deployment, though the recommendation is always advisory
  (FR-10.2). Deployment is never automatic (FR-11.2, ADL-23):
  `confirm_deployment(..., confirmed: bool)` is the only function in the
  codebase that writes the atomically-updated active-version pointer file,
  and every decision — deployed or rejected — persists to a new
  `model_deployments` audit table. `rollback_to_last_known_good()`
  implements NFR-5.3's automatic fallback by walking that table backwards
  for the most recent successfully-deployed version.
- `aiperfopt.bootstrap.bootstrap()` — the single entrypoint that wires
  config → logging → storage (including schema creation) in the required
  order; every future entrypoint (API server, PySide6 app, CLI, tests)
  should call this rather than constructing these independently.
- **API Layer** (`aiperfopt.api`) — a FastAPI app (`create_app()`,
  ADL-16) that is the only access path to backend functionality
  (FR-12.4): REST endpoints wrap every existing `storage.reader`
  function for historical queries (FR-12.2), Phase 8's retraining
  pipeline for a trigger/confirm/rollback endpoint trio (FR-11.x exposed
  over HTTP, still human-confirmed per ADL-23 — the client round-trips
  the exact comparison it was shown so nothing can be silently
  substituted between review and confirmation), and OpenAPI docs are
  auto-generated (FR-12.3, `/docs`). The actual substance of this phase
  is the **live-inference loop** (`LiveInferenceLoop`) — a dedicated
  thread (ADL-18's pattern) that, for the first time in this codebase,
  runs feature → forecast → contention → diagnosis → recommendation
  continuously rather than as a one-shot call, streaming each snapshot
  over WebSocket (`/ws/live`, FR-12.1). Two small but real pieces of new
  infrastructure make this possible:
  `forecasting.resolve_deployed_forecasters()` reconstructs a
  model_version's forecasters purely from persisted evaluation history
  (no in-memory training object needed, since live inference runs in a
  different process/moment than training), and
  `contention.build_live_contention_input_row()` is the live-inference
  counterpart to Phase 5's training-only `build_contention_dataset`.
  Every stage of a live cycle degrades gracefully rather than erroring
  (NFR-11.1/11.2) — no history yet, no model deployed yet, and a model
  load failure triggers Phase 8's `rollback_to_last_known_good()`
  automatically (NFR-5.3) before giving up on that cycle. Run with
  `python3 -m aiperfopt.api.server`.
- **Reporting (Section 7.1 item 14, small Phase 10 addition)** —
  `GET /reporting/predicted-vs-actual` (`aiperfopt.reporting.predicted_vs_actual`,
  exposed via `aiperfopt.api.routers.reporting`) joins forecast
  predictions against the actual metric value nearest each
  `target_timestamp`, powering FR-13.3's comparison view.
- **Frontend** (`aiperfopt.frontend`, Phase 10, ADL-16, FR-13.1-13.3) — a
  PySide6 desktop client, a pure API consumer (FR-13.2): nothing under
  this package imports storage or model code. `ApiClient` wraps every
  REST endpoint above; `LiveWebSocketClient` streams `/ws/live` into the
  Qt event loop via `QWebSocket`. General Mode (`DashboardWidget`) and
  Developer Mode (`DeveloperPanel`) render the *same* `LiveStateOut`
  payload — the mode toggle is a pure rendering distinction, not a
  separate data path. `PredictedVsActualChart`/`PredictedVsActualTab`
  render Section 6's flagship comparison view against the reporting
  endpoint above. `RetrainingPanel` exposes Phase 8/9's
  run/confirm/reject/rollback flow, with every call routed through
  `workers.run_in_background()` (`QThreadPool`) so the multi-minute
  `/retraining/run` call never freezes the window. Run with
  `python3 -m aiperfopt.frontend.main` (the backend must be reachable,
  though not necessarily already running — the WebSocket client retries
  indefinitely per NFR-11.2).
- **Reporting / Analytics** (`aiperfopt.reporting`, Phase 11, Section 7.1
  item 14, FR-14.1/FR-14.2, ADL-22) — `build_report()`
  (`aiperfopt.reporting.summary`) produces an exportable daily/weekly (or
  custom-window) summary combining system-metrics context with
  prediction accuracy (FR-14.2's explicit requirement, reusing
  `predicted_vs_actual.join_predicted_vs_actual` plus
  `forecasting.evaluation.compute_mae_rmse`/`compute_directional_accuracy`
  — the same metrics Section 11's walk-forward validation reports,
  computed here over live-observed predictions), contention, diagnosis,
  and recommendation summaries. Scoped strictly to the three flagship
  pillars (ADL-22), not generic dashboarding — system-metrics summaries
  are included only as interpretive context for the other three.
  `GET /reporting/summary?period=daily|weekly` (or `?start=&end=`)
  exposes it; `?download=true` returns the same JSON as a downloadable
  attachment (FR-14.1's "exportable," without inventing a second export
  format — Section 18 defers that kind of detail, and JSON-over-REST is
  the project's existing default everywhere else). The Phase 10
  predicted-vs-actual join moved from the API router into this package
  (`reporting/predicted_vs_actual.py`) once this became its second
  consumer, per that router's own docstring flagging this as the
  intended landing spot. A `ReportingTab` in the Phase 10 frontend
  (available in both General and Developer Mode, unlike Retraining)
  renders the report.

Every Section 7.1 subsystem is now at least minimally implemented.

## Setup

Requires Python ≥ 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

Copy `config/.env.example` to `.env` and/or create a user override YAML
file (any subset of `config/default.yaml`'s structure). Load order,
highest precedence first:

1. Environment variables (`AIPERFOPT_SECTION__FIELD=value`)
2. `.env` file
3. User override YAML (pass its path to `load_settings()`)
4. `config/default.yaml` (packaged defaults)

```python
from aiperfopt.config import load_settings

settings = load_settings("path/to/my_overrides.yaml")  # or load_settings() for defaults+env only
```

## Running tests

```bash
pytest
```

## Project layout

```
src/aiperfopt/
├── common/       # Shared exception hierarchy
├── config/       # Layered configuration (Phase 0)
├── applog/       # Structured JSON logging (Phase 0)
├── storage/      # Data Storage subsystem (Phase 0 + Phase 1)
│   ├── engine.py            # Connection bootstrap (Phase 0)
│   ├── schema.py             # Table definitions (Phase 1, FR-3.1)
│   ├── schema_management.py  # create_all_tables / verify_schema
│   ├── records.py            # Typed row dataclasses
│   ├── writer.py              # Insert API (single + batch)
│   ├── reader.py              # Chronological range-query API (FR-3.3)
│   └── rollup.py              # Downsampling + retention (FR-3.2)
├── monitoring/   # Monitoring Engine (Phase 2) — producer side
│   ├── collectors.py          # System-metric collection + degradation (FR-1.1, FR-1.4)
│   ├── process_tracker.py     # Per-process polling + launch/terminate detection (FR-1.3)
│   ├── queue_items.py         # Shared Monitoring -> Ingestion queue item contract
│   └── engine.py               # Threaded MonitoringEngine (FR-1.2, ADL-18)
├── ingestion/    # Data Ingestion (Phase 2) — consumer side
│   └── consumer.py             # Batched, ordered storage writes (FR-2.1-2.3)
├── features/     # Feature Engineering (Phase 3)
│   ├── config.py               # FeatureEngineeringConfig (metrics, windows, tolerances)
│   ├── engine.py                # Pure computation — the FR-4.2 single source of truth
│   └── storage_bridge.py        # Storage-aware wrappers (live vector, backfill)
├── forecasting/  # Forecasting Models + Evaluation Harness (Phase 4)
│   ├── dataset.py                # Supervised dataset: feature matrix + forward-looking label
│   ├── baselines.py               # Forecaster protocol + 3 mandatory baselines (Section 10.8)
│   ├── model.py                    # MetricForecaster — the LightGBM-backed model (ADL-13)
│   ├── persistence.py               # save/load trained model artifacts
│   ├── deployment.py                 # load_deployed_forecaster() — reconstruct whichever forecaster won
│   ├── evaluation.py                  # Walk-forward validation harness (Section 11, FR-10.1)
│   └── training_pipeline.py            # Orchestrator: train_and_evaluate_all()
├── synthetic.py  # Scenario generator (Section 10.3 Phase 1); shared by Forecasting/Contention/Anomaly
├── anomaly/      # Anomaly Detection (Phase 5)
│   ├── detector.py               # Isolation Forest, normalized [0,1] anomaly score (FR-7.1)
│   ├── persistence.py             # save/load detector artifacts (joblib)
│   └── notifications.py            # Distinct informational notification on significant deviation (FR-7.2)
├── contention/   # Contention Prediction (Phase 5)
│   ├── config.py                  # ContentionLabelingConfig — training-label thresholds
│   ├── dataset.py                  # build_contention_dataset() — forecast+state+anomaly join (FR-6.1)
│   ├── model.py                     # ContentionClassifier — logistic regression + coefficient attribution
│   ├── processes.py                  # Contributing-processes lookup (FR-6.2)
│   ├── prediction.py                  # Live inference: anomaly dampening + record assembly
│   ├── persistence.py                  # save/load classifier artifacts (joblib)
│   ├── evaluation.py                    # Walk-forward precision/recall/F1
│   └── training_pipeline.py              # Orchestrator: train_contention_and_anomaly_pipeline()
├── diagnosis/    # Diagnosis / Explainable AI (Phase 6)
│   ├── taxonomy.py                 # DiagnosisCategory (Section 12's 8 values) + resolve_diagnosis_category()
│   ├── explainer.py                  # Tiered lightweight/SHAP forecast attribution (Section 10.5)
│   ├── text.py                        # Technical (Developer Mode) + plain-language (General Mode) templates
│   └── service.py                      # Orchestrator: diagnose_contention_event()
├── recommendation/    # Recommendation Engine (Phase 7)
│   ├── catalog.py                 # RecommendationCatalog + load_recommendation_catalog()
│   └── service.py                  # Orchestrator: generate_recommendations()
├── retraining/    # Model Retraining (Phase 8)
│   ├── real_usage_data.py         # Chronological train/holdout split of accumulated real history
│   ├── active_version.py           # Atomic active-model-version pointer file read/write
│   ├── comparison.py                # Candidate-vs-active regression detection (FR-10.2)
│   ├── deployment.py                 # confirm_deployment() / rollback_to_last_known_good() (FR-11.2/11.3)
│   └── pipeline.py                    # Orchestrator: run_retraining_on_real_usage_data()
├── api/    # API Layer (Phase 9)
│   ├── app.py                    # create_app() factory; lifespan wires shutdown, never auto-starts
│   ├── server.py                  # python -m aiperfopt.api.server — the real uvicorn entrypoint
│   ├── live_state.py               # LiveInferenceLoop + LiveState — the running live pipeline (FR-12.1)
│   ├── connection_manager.py        # WebSocket broadcast fan-out
│   ├── dependencies.py               # FastAPI Depends() providers over AppContext/live loop
│   ├── schemas.py                     # Pydantic response/request models (1:1 with storage records)
│   └── routers/
│       ├── system.py                     # /health, /config, /monitoring/*, /live/status|start|stop
│       ├── history.py                     # GET wrappers over every storage.reader function (FR-12.2)
│       ├── retraining.py                   # /retraining/run|confirm|rollback|active-version
│       ├── live.py                          # GET /live/state, WS /ws/live
│       └── reporting.py                      # Thin wrapper over aiperfopt.reporting (below)
├── reporting/    # Reporting / Analytics (Phase 11, Section 7.1 item 14)
│   ├── predicted_vs_actual.py     # join_predicted_vs_actual() — nearest-timestamp join (FR-13.3)
│   └── summary.py                  # build_report() — system/prediction/contention/diagnosis/recommendation (FR-14.1-14.2)
├── frontend/    # Frontend (Phase 10) — pure API client (FR-13.2)
│   ├── config.py                  # FrontendSettings (host/port; the one locally-held setting)
│   ├── api_client.py               # ApiClient — sync httpx wrapper over every REST endpoint
│   ├── live_client.py               # LiveWebSocketClient — QWebSocket streaming of /ws/live
│   ├── workers.py                    # run_in_background() — QThreadPool helper for slow calls
│   ├── formatting.py                  # Pure display-formatting helpers (Qt-free, unit-testable)
│   ├── main_window.py                  # MainWindow — General/Developer mode toggle, tab assembly
│   ├── main.py                          # python -m aiperfopt.frontend.main entrypoint
│   └── widgets/
│       ├── dashboard.py                     # General Mode (Section 4)
│       ├── developer_panel.py                # Developer Mode (Section 4) — same payload, more fields
│       ├── predicted_vs_actual.py             # QtCharts predicted-vs-actual series (FR-13.3)
│       ├── predicted_vs_actual_tab.py          # Metric/lookback controls driving the chart above
│       ├── reporting_tab.py                     # Renders /reporting/summary (Phase 11, both modes)
│       └── retraining_panel.py                    # Developer Mode retraining run/confirm/reject/rollback
├── pipeline.py   # Wires Monitoring + Ingestion together (not auto-started by bootstrap)
└── bootstrap.py  # config -> logging -> storage wiring
tests/            # Mirrors src/ layout, one test module per subsystem
config/
├── default.yaml           # Packaged configuration defaults
├── .env.example            # Documents available environment overrides
└── recommendations.yaml     # Recommendation Engine catalog (Phase 7, NFR-12.1)
```

## Roadmap

See `PROJECT_SPEC.md` Section 20 for the long-term roadmap. Near-term
implementation order (agreed roadmap for this development effort):

0. ~~Project Foundation~~ ✅
1. ~~Data Storage (schema, rollup/retention)~~ ✅
2. ~~Monitoring Engine + Data Ingestion~~ ✅
3. ~~Feature Engineering~~ ✅
4. ~~Forecasting Models + Evaluation Harness~~ ✅
5. ~~Contention Prediction + Anomaly Detection~~ ✅
6. ~~Diagnosis / Explainable AI~~ ✅
7. ~~Recommendation Engine~~ ✅
8. ~~Model Retraining~~ ✅
9. ~~API Layer~~ ✅
10. ~~Frontend~~ ✅
11. ~~Reporting / Analytics~~ ✅

Every Section 7.1 subsystem is now at least minimally implemented — see
`PROJECT_SPEC.md` Section 20 for what's next beyond V1's near-term order
above (statistical calibration, multi-horizon sequence models,
cross-platform support, etc.).

## Running the full Phase 0-6 pipeline end-to-end

```python
from aiperfopt.bootstrap import bootstrap
from aiperfopt.features import FeatureEngineeringConfig
from aiperfopt.forecasting import train_and_evaluate_all
from aiperfopt.contention import ContentionLabelingConfig, train_contention_and_anomaly_pipeline

ctx = bootstrap()
fconfig = FeatureEngineeringConfig()

# Phase 4: train + evaluate forecasters, save winning artifacts.
forecast_summary = train_and_evaluate_all(
    ctx.storage_engine, fconfig, ctx.settings.forecasting, model_version="v1",
    target_metrics=["cpu_percent", "mem_percent"],
)
short_outcomes = {o.metric_name: o for o in forecast_summary.outcomes if o.horizon_label == "short"}

# Phase 5: train anomaly detector + contention classifier on top of the deployed forecasters.
summary = train_contention_and_anomaly_pipeline(
    ctx.storage_engine, fconfig, ctx.settings.forecasting, ctx.settings.contention,
    ContentionLabelingConfig(), short_outcomes, model_version="v1",
)
print(summary.walk_forward_summary, summary.holdout_metrics)
```

Or run the live monitor separately (no ML involved, just real data collection):

```python
from aiperfopt.pipeline import create_monitoring_pipeline

pipeline = create_monitoring_pipeline(ctx)
pipeline.start()
# ... runs in background threads ...
pipeline.stop()
```

## Diagnosing a single contention event (Phase 6)

Diagnosis consumes a `ContentionPredictionRecord` (from `predict_contention`,
above) plus the same-cycle engineered feature vector and the deployed
forecasters — it does not need the live classifier itself:

```python
from aiperfopt.contention import load_contention_classifier, predict_contention
from aiperfopt.diagnosis import diagnose_contention_event
from aiperfopt.forecasting import load_deployed_forecaster

classifier = load_contention_classifier(ctx.settings.forecasting.models_dir, "v1")
deployed_forecasters = {
    name: load_deployed_forecaster(outcome, ctx.settings.forecasting.models_dir, "v1")
    for name, outcome in short_outcomes.items()
}

# feature_row is the full engineered vector for this cycle, e.g. from
# aiperfopt.features.compute_and_store_live_feature_vector(...).feature_values
contention_record = predict_contention(
    classifier, ctx.storage_engine, feature_row, as_of=feature_row["timestamp"],
    decision_threshold=ctx.settings.contention.decision_threshold,
    anomaly_dampening_factor=ctx.settings.contention.anomaly_dampening_factor,
    contributing_processes_top_n=ctx.settings.contention.contributing_processes_top_n,
    model_version="v1",
)

diagnosis = diagnose_contention_event(
    ctx.storage_engine, contention_record, feature_row, deployed_forecasters,
    ctx.settings.contention, ctx.settings.diagnosis, model_version="v1",
)
if diagnosis is not None:
    print(diagnosis.category, diagnosis.plain_language_explanation)
```

## Generating a recommendation from a diagnosis (Phase 7)

`generate_recommendations` needs the diagnosis's persisted `id` (populated
on the record `diagnose_contention_event` returns) — it's a pure
`category -> curated actions` lookup, nothing else:

```python
from aiperfopt.recommendation import generate_recommendations, load_recommendation_catalog

catalog = load_recommendation_catalog()  # config/recommendations.yaml
if diagnosis is not None:
    recommendation = generate_recommendations(ctx.storage_engine, diagnosis, catalog)
    if recommendation is not None:  # None for the anomaly category (ADL-36) — informational only
        for action in recommendation.actions:
            print(action["title"])
```

## Retraining on accumulated real-usage data (Phase 8)

Requires the live monitor to have already written history into
`raw_system_metrics` — there's nothing to retrain on with a freshly
bootstrapped database. Reuses the exact same Phase 4/5 training pipelines
as the initial synthetic bootstrap, just against real accumulated data
and with a comparison-against-what's-currently-active step before
anything is deployed:

```python
from aiperfopt.features import FeatureEngineeringConfig
from aiperfopt.contention import ContentionLabelingConfig
from aiperfopt.retraining import (
    run_retraining_on_real_usage_data,
    confirm_deployment,
    get_active_model_version,
    rollback_to_last_known_good,
)

candidate = run_retraining_on_real_usage_data(
    ctx.storage_engine, FeatureEngineeringConfig(), ctx.settings.forecasting,
    ctx.settings.contention, ContentionLabelingConfig(), ctx.settings.retraining,
    new_model_version="v2", target_metrics=["cpu_percent", "mem_percent"],
)
print(candidate.comparison.recommendation, candidate.comparison.reasons)

# Deployment is never automatic (FR-11.2, ADL-23) -- confirmed is a
# required, explicit argument with no default:
confirm_deployment(
    ctx.storage_engine, ctx.settings.retraining, "v2", candidate.comparison, confirmed=True,
)
print("active version:", get_active_model_version(ctx.settings.retraining))

# NFR-5.3's automatic fallback, if a deployed model ever fails at inference time:
# rollback_to_last_known_good(ctx.storage_engine, ctx.settings.retraining, reason="...")
```

## Running the API server (Phase 9)

```bash
python3 -m aiperfopt.api.server
```

This auto-starts Monitoring and the live-inference loop on startup
(feature → forecast → contention → diagnosis → recommendation, ticking
every `api.live_inference_interval_seconds`). Then:

- `http://127.0.0.1:8000/docs` — auto-generated OpenAPI docs (FR-12.3)
- `ws://127.0.0.1:8000/ws/live` — live pipeline state, pushed every cycle (FR-12.1)
- `GET /live/state` — the same snapshot, as a REST fallback
- `GET /metrics/system`, `/features`, `/forecasts`, `/contention`, `/diagnoses`,
  `/recommendations`, `/evaluations`, `/deployments` — historical range
  queries (`?start=`, `?end=`, `?limit=`), one per subsystem (FR-12.2)
- `POST /retraining/run`, `POST /retraining/confirm`, `POST /retraining/rollback`,
  `GET /retraining/active-version` — Phase 8 exposed over HTTP

On a freshly bootstrapped database nothing is deployed yet, so
`/live/state` reports metrics/features only until a model is trained and
confirmed:

```python
import httpx

client = httpx.Client(base_url="http://127.0.0.1:8000")

r = client.post("/retraining/run", json={
    "new_model_version": "v1", "data_source": "real_usage",
    "target_metrics": ["cpu_percent", "mem_percent"],
})
candidate = r.json()
print(candidate["comparison"]["recommendation"])

# Deployment is never automatic (FR-11.2, ADL-23) -- the client round-trips
# the exact comparison it was shown, and "confirmed" has no default:
client.post("/retraining/confirm", json={
    "candidate_version": candidate["model_version"],
    "comparison": candidate["comparison"],
    "confirmed": True,
})
```

Building a FastAPI app without the real server process (e.g. for tests)
never starts any background thread as a side effect:

```python
from aiperfopt.api import create_app
from aiperfopt.bootstrap import bootstrap

app = create_app(bootstrap(), auto_start=False)  # dormant until /monitoring/start, /live/start
```

## Running the frontend (Phase 10)

With the API server (above) already running, or about to be — the
WebSocket client retries indefinitely until it can connect, so start
order doesn't matter:

```bash
python3 -m aiperfopt.frontend.main
```

This opens the main window: a connection-status indicator, a
General/Developer mode toggle, and four tabs (Live, Predicted vs.
Actual, Reporting, Retraining — the last hidden in General Mode). "Start
monitoring" and "Start live inference" drive the same
`/monitoring/*`/`/live/*` endpoints Phase 9 exposes; once a model is
confirmed deployed (see the retraining example above, or use the
Retraining tab itself), the Live tab starts showing real forecasts,
contention probability, diagnosis, and recommendations as they arrive
over `/ws/live`.

To point the frontend at a non-default backend address:

```bash
AIPERFOPT_FRONTEND__API_HOST=127.0.0.1 AIPERFOPT_FRONTEND__API_PORT=8000 \
    python3 -m aiperfopt.frontend.main
```

Frontend widget/integration tests (`tests/frontend/`) run against a real
backend on a real local port (`tests/frontend/server_fixture.py`), not
an in-process shortcut — `pytest-qt`'s `qtbot` fixture drives the Qt
event loop, and `QT_QPA_PLATFORM=offscreen` lets the whole suite run
headless:

```bash
QT_QPA_PLATFORM=offscreen pytest tests/frontend/ -q
```

## Generating a report (Phase 11)

```python
import httpx

client = httpx.Client(base_url="http://127.0.0.1:8000")

# Named period ("daily"/"weekly"), ending now:
r = client.get("/reporting/summary", params={"period": "weekly"})
report = r.json()
print(report["prediction_accuracy"])   # FR-14.2: MAE/RMSE/directional accuracy per metric+horizon
print(report["diagnosis"]["counts_by_category"])  # zero-filled for every Section 12 category

# Or an explicit window:
r = client.get("/reporting/summary", params={"start": 1_700_000_000, "end": 1_700_086_400})

# FR-14.1's "exportable": the same body as a downloadable attachment.
r = client.get("/reporting/summary", params={"period": "daily", "download": True})
# r.headers["content-disposition"] == 'attachment; filename="aiperfopt_report_..._....json"'
```

`aiperfopt.reporting.summary.build_report()` is the underlying pure
function (no FastAPI/HTTP involved) if you want a report from a script
or notebook without going through the API layer.

