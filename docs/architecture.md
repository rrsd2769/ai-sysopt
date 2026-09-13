# Architecture (Phase 10.1)

Source-of-truth diagram for Review 2's System Design and Architecture
requirement. Rendered from Mermaid so it stays a diffable text file, not a
one-off exported image (`docs/plans/2026-09-12-presentable-predictive-monitor.md`,
Phase 10 Deliverable 1).

## Pipeline

Every stage below is a real module under `src/aiperfopt/` (see
[`docs/phase10-module-table.md`](phase10-module-table.md) for
responsibilities, source paths, and test counts). Solid arrows are the
per-cycle live-inference path (`aiperfopt.api.live_state.LiveInferenceLoop._tick`,
one cycle every `api.live_inference_interval_seconds`); dashed arrows are
the two loops that run on their own separate cadence rather than every
tick.

```mermaid
flowchart TD
    subgraph Producer["Producer (background threads)"]
        MON["Monitoring Engine<br/>system + process pollers"]
        ING["Data Ingestion<br/>batched ordered writes"]
        MON -->|shared queue| ING
    end

    ING --> STORE[("Data Storage<br/>SQLite: raw metrics, process rows,<br/>features, forecasts, diagnoses")]

    STORE --> FEAT["Feature Engineering<br/>system + process lag/rolling windows<br/>(single source of truth: live == train)"]

    FEAT --> FORECAST["Forecasting<br/>LightGBM vs. 3 baselines"]
    FEAT --> ANOM["Anomaly Detection<br/>Isolation Forest"]
    FORECAST --> CONT["Contention Prediction<br/>logistic classifier"]
    ANOM --> CONT

    CONT --> DIAG["Diagnosis / Explainable AI<br/>taxonomy + SHAP/lightweight attribution"]
    DIAG --> REC["Recommendation Engine<br/>category to curated actions"]

    FORECAST --> API["API Layer<br/>FastAPI: REST + WebSocket"]
    CONT --> API
    DIAG --> API
    REC --> API
    STORE --> API

    API -->|WS /ws/live, REST /live/state| FRONT["Frontend<br/>PySide6: dashboard, live chart,<br/>Predicted vs. Actual, Reporting, Retraining"]

    STORE -.->|periodic, on demand| REPORT["Reporting / Analytics<br/>predicted-vs-actual join,<br/>period summaries"]
    REPORT -.-> API

    STORE -.->|accumulated real usage| RETRAIN["Model Retraining<br/>candidate vs. active comparison,<br/>quality gate, confirm/rollback"]
    RETRAIN -.->|deploy on confirm| FORECAST
    RETRAIN -.->|deploy on confirm| CONT
    RETRAIN -.-> ANOM

    REPLAY["aiperfopt replay<br/>(Phase 8, demo-only)"] -.->|writes to a separate<br/>replay database| STORE
```

## Notes that don't fit in the picture

- **Replay is not a twelfth module.** It reuses the same Storage/Feature/Forecasting/Contention path against a separate replay database (`<primary>-replay.<ext>`), pacing writes from a previously recorded session rather than a live poller. The API server is told it's serving replayed data via `StorageSettings.is_replay` (`AIPERFOPT_STORAGE__IS_REPLAY=true`), which the Frontend surfaces as the `REPLAY` readiness state.
- **`/readiness` is deliberately outside the live-inference loop.** It answers "can this system predict right now, and why not" from `aiperfopt.readiness.assess_readiness`, callable with no server running at all (`aiperfopt doctor`) — it is not one of the boxes above because it inspects Storage and the active model pointer directly rather than participating in the per-cycle tick.
- **Retraining never auto-deploys.** `run_retraining_on_real_usage_data` produces a candidate and a comparison against the currently active version; `confirm_deployment` requires an explicit `confirmed=True` from an operator (CLI, API, or the Retraining tab) before Forecasting/Contention/Anomaly's active artifacts change.
