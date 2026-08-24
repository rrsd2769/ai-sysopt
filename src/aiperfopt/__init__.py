"""AI System Performance Optimizer — predictive macOS system health monitor.

See PROJECT_SPEC.md for the authoritative specification. This package is
organized to mirror the subsystem map in Section 7.1:

    config       -> Configuration (Section 7.1, item 15)
    applog       -> Logging (item 16)
    storage      -> Data Storage (item 3)
    monitoring   -> Monitoring Engine (item 1)          [Phase 2]
    ingestion    -> Data Ingestion / Collector (item 2)  [Phase 2]
    features     -> Feature Engineering (item 4)         [Phase 3]
    forecasting  -> Forecasting Models (item 5)           [Phase 4]
    evaluation   -> Model Evaluation & Validation (item 10) [Phase 4]
    contention   -> Contention Prediction (item 6)         [Phase 5]
    anomaly      -> Anomaly Detection (item 7)             [Phase 5]
    diagnosis    -> Diagnosis / Explainable AI (item 8)    [Phase 6]
    recommend    -> Recommendation Engine (item 9)         [Phase 7]
    retraining   -> Model Retraining (item 11)             [Phase 8]
    api          -> API Layer (item 12)                    [Phase 9]
    frontend     -> Frontend (item 13)                     [Phase 10]
    reporting    -> Reporting / Analytics (item 14)         [Phase 11]

Only ``config``, ``applog``, ``storage``, and ``common`` exist as of
Phase 0.
"""

__version__ = "0.1.0"
