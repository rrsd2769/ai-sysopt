"""AI System Performance Optimizer — predictive macOS system health monitor.

This package is organized by subsystem:

    config, applog, common -> configuration, logging, shared helpers
    storage                -> SQLite schema, writer, reader, retention
    monitoring             -> collectors, ingestion queue, batched writer
    features               -> feature engineering and preprocessing
    forecasting            -> LightGBM forecaster and statistical baselines
    anomaly                -> Isolation Forest anomaly detector
    contention             -> contention classifier
    insight                -> diagnosis (SHAP) and recommendations
    lifecycle              -> deployment gate, retraining, rollback
    evaluation             -> ablation and the three-model comparison
    reporting              -> summaries and predicted-vs-actual
    replay                 -> record and replay a session for demos
    api                    -> FastAPI server and live-inference loop
    web                    -> browser dashboard served by the API
"""

__version__ = "0.1.0"
