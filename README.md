# AI System Performance Optimizer

Predictive, explainable system health monitor for **Windows and macOS**.
It watches your machine, forecasts where CPU and memory are heading,
predicts resource contention before you feel it, explains why, and
suggests what to do about it.

> **Repair in progress.** This project is being brought back to a working
> state against a measured diagnosis. See
> [`docs/plans/2026-08-24-make-it-actually-predict.md`](docs/plans/2026-08-24-make-it-actually-predict.md)
> for what is broken, what is being fixed, and in what order. Phase 0
> (foundations) is complete; the model quality work is not. **Do not yet
> trust the predictions this produces** — the forecasters currently lose
> to naive baselines and the anomaly detector saturates. That is Phases
> 3–6.

## Quickstart

Requires **Python 3.12** (3.13+ is not yet supported by parts of the ML
stack; 3.11 and below lack features this code uses).

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Then, on either platform:

```bash
aiperfopt doctor
```

That reports your environment, every resolved path, and whether the
system can currently predict anything — and tells you the next command to
run. From a standing start:

```bash
aiperfopt collect --minutes 15
```

```bash
aiperfopt train --source synthetic --deploy
```

```bash
aiperfopt run
```

`run` starts the backend and opens the desktop UI against it. Every
command also works as `python -m aiperfopt <command>`.

## Commands

| Command | What it does |
|---|---|
| `aiperfopt doctor` | Environment, resolved paths, readiness, and the next step |
| `aiperfopt collect --minutes N` | Collect metrics for a fixed period, no server |
| `aiperfopt train [--source synthetic\|real] [--deploy]` | Train a candidate model; `--deploy` activates it |
| `aiperfopt serve` | API server with monitoring and live inference |
| `aiperfopt ui` | Desktop UI against an already-running backend |
| `aiperfopt run` | Backend + UI together |

Add `--log-level DEBUG` to any command for verbose output, or
`--config path/to/override.yaml` to layer in your own settings.

## Where your data lives

Everything the application writes — the SQLite database, trained model
artifacts, logs, and the active-version pointer — lives under one
application home:

| Platform | Default location |
|---|---|
| Windows | `%LOCALAPPDATA%\aiperfopt` |
| macOS | `~/Library/Application Support/aiperfopt` |
| Linux | `$XDG_DATA_HOME/aiperfopt` or `~/.local/share/aiperfopt` |

Override it by setting `AIPERFOPT_HOME`. This matters: these paths used
to resolve against the current working directory, so launching from two
different folders silently produced two separate empty databases.
`aiperfopt doctor` always prints exactly which files are in use.

## Configuration

Layered, highest priority first:

1. Environment variables — prefix `AIPERFOPT_`, nested delimiter `__`
   (e.g. `AIPERFOPT_MONITORING__POLLING_INTERVAL_SECONDS=2.0`)
2. A `.env` file
3. A user override YAML — `--config`, or `AIPERFOPT_CONFIG_FILE`
4. The packaged defaults (`src/aiperfopt/resources/default.yaml`)

The defaults ship *inside* the package so they are present in a built
wheel and a PyInstaller bundle. Relative paths in any layer resolve
against `AIPERFOPT_HOME`, never the working directory.

## Running tests

```bash
pytest --ignore=tests/frontend -q
```

The backend suite is currently **384 passing**. The Qt frontend suite
segfaults (`exit 139`) on a current environment — a known defect
scheduled for Phase 8 of the repair plan, tracked in CI as a
non-blocking job rather than hidden behind a skip:

```bash
QT_QPA_PLATFORM=offscreen pytest tests/frontend -q
```

Be aware that a green backend suite does **not** currently imply the
models work: the suite tests plumbing, not answer quality. Phase 2 of the
repair plan labels every test accordingly.

## Continuous integration

`.github/workflows/ci.yml` runs the backend suite on **both
`windows-latest` and `macos-latest`**, plus a job asserting that a built
wheel actually contains its YAML configuration. Windows is a first-class
target: several defects are Windows-only and silent, so the suite has to
really run there.

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

> **Superseded.** `aiperfopt train --deploy` does all of this. The code
> below is kept as a description of what the pipeline does internally,
> not as instructions anyone should have to follow.

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

catalog = load_recommendation_catalog()  # aiperfopt/resources/recommendations.yaml
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

