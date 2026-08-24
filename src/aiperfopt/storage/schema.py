"""
Data Storage schema (FR-3.1, FR-3.3).

All tables are plain SQLAlchemy Core ``Table`` objects against a shared
``MetaData`` (ADL-17 — Core, not the ORM). Five persistence categories are
kept as distinct table groups per FR-3.1 ("raw metric samples, computed
features, model predictions, evaluation results, and structured logs in
separate tables/schemas"):

    raw metric samples   -> raw_system_metrics, raw_process_metrics, process_events
    computed features    -> features
    model predictions    -> forecast_predictions, contention_predictions
    evaluation results   -> evaluations
    structured logs      -> app_logs

Plus one supporting table not in FR-3.1's list:

    raw_system_metrics_rollup  -> downsampled system metrics (FR-3.2)

Design notes
------------
- Every table that participates in chronological queries has an explicit
  ``Index`` on its timestamp column (FR-3.3: "efficient chronological
  range queries (indexed on timestamp)").
- Timestamps are stored as UTC Unix epoch floats (``Float``), not SQLite's
  string-based datetime affinity. This keeps range comparisons a plain
  numeric ``BETWEEN``/``<``/``>`` (fast, index-friendly) and sidesteps
  timezone-string parsing entirely — every producer in this codebase uses
  ``time.time()`` and every consumer compares against the same float
  epoch, so there is exactly one representation to reason about.
- ``JSON`` columns are used deliberately wherever the payload's shape
  belongs to a subsystem that hasn't been built yet (feature vectors:
  Phase 3; contributing-factor breakdowns: Phase 5/6) or is inherently
  variable-width (evaluation ``extra`` metrics). This avoids coupling the
  Phase 1 schema to decisions later phases haven't made yet, consistent
  with NFR-13.1 (new modules addable with minimal modification to
  existing components).
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)

# A shared naming convention keeps auto-generated constraint/index names
# predictable, which matters if a future version introduces Alembic
# migrations (ADL-17 notes SQLAlchemy Core "protects future migration").
_NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=_NAMING_CONVENTION)

# ---------------------------------------------------------------------------
# Raw metric samples (Section 9 core monitoring metric set)
# ---------------------------------------------------------------------------

raw_system_metrics = Table(
    "raw_system_metrics",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", Float, nullable=False),
    # CPU
    Column("cpu_percent", Float, nullable=False),
    Column("cpu_count_logical", Integer, nullable=True),
    # Memory (physical + virtual/swap)
    Column("mem_total", Integer, nullable=True),
    Column("mem_used", Integer, nullable=True),
    Column("mem_available", Integer, nullable=True),
    Column("mem_percent", Float, nullable=False),
    Column("swap_total", Integer, nullable=True),
    Column("swap_used", Integer, nullable=True),
    Column("swap_percent", Float, nullable=True),
    # Disk I/O activity (cumulative counters as reported by the OS)
    Column("disk_read_bytes", Integer, nullable=True),
    Column("disk_write_bytes", Integer, nullable=True),
    Column("disk_read_count", Integer, nullable=True),
    Column("disk_write_count", Integer, nullable=True),
    # Network activity (cumulative counters)
    Column("net_bytes_sent", Integer, nullable=True),
    Column("net_bytes_recv", Integer, nullable=True),
    Column("net_packets_sent", Integer, nullable=True),
    Column("net_packets_recv", Integer, nullable=True),
    # System load / scheduling pressure
    Column("load_avg_1", Float, nullable=True),
    Column("load_avg_5", Float, nullable=True),
    Column("load_avg_15", Float, nullable=True),
    Column("thread_count_total", Integer, nullable=True),
    # Battery / power (IOKit, user-space per Section 5)
    Column("battery_percent", Float, nullable=True),
    Column("power_plugged", Boolean, nullable=True),
)
Index("ix_raw_system_metrics_timestamp", raw_system_metrics.c.timestamp)

raw_process_metrics = Table(
    "raw_process_metrics",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", Float, nullable=False),
    Column("pid", Integer, nullable=False),
    Column("name", String(256), nullable=False),
    Column("cpu_percent", Float, nullable=False),
    Column("memory_rss", Integer, nullable=True),
    Column("memory_vms", Integer, nullable=True),
    Column("memory_percent", Float, nullable=True),
    Column("num_threads", Integer, nullable=True),
    # macOS/POSIX "nice" value — the available scheduling information
    # accessible from user-space (Section 5's kernel-level exclusion note).
    Column("priority", Integer, nullable=True),
    Column("status", String(32), nullable=True),
    Column("create_time", Float, nullable=True),
)
Index("ix_raw_process_metrics_timestamp", raw_process_metrics.c.timestamp)
Index("ix_raw_process_metrics_pid", raw_process_metrics.c.pid)

process_events = Table(
    "process_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", Float, nullable=False),
    Column("pid", Integer, nullable=False),
    Column("name", String(256), nullable=False),
    # "launch" | "terminate" — FR-1.3.
    Column("event_type", String(16), nullable=False),
)
Index("ix_process_events_timestamp", process_events.c.timestamp)

# ---------------------------------------------------------------------------
# Downsampled / rolled-up system metrics (FR-3.2)
# ---------------------------------------------------------------------------

raw_system_metrics_rollup = Table(
    "raw_system_metrics_rollup",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    # Start of the aggregation bucket, e.g. the top of the hour it covers.
    Column("bucket_start", Float, nullable=False),
    Column("bucket_seconds", Integer, nullable=False),
    Column("sample_count", Integer, nullable=False),
    # {"cpu_percent": {"avg": .., "min": .., "max": ..}, "disk_read_bytes": {"sum": ..}, ...}
    Column("aggregates", JSON, nullable=False),
)
Index("ix_raw_system_metrics_rollup_bucket_start", raw_system_metrics_rollup.c.bucket_start)

# ---------------------------------------------------------------------------
# Computed features (Feature Engineering output, Phase 3)
# ---------------------------------------------------------------------------

features = Table(
    "features",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    # The timestamp this feature vector describes (i.e. the "as of" time
    # used for lag/rolling-window computation).
    Column("timestamp", Float, nullable=False),
    # When the row was written — differs from `timestamp` during batch
    # backfill/retraining feature computation vs. live inference.
    Column("computed_at", Float, nullable=False),
    # Lets feature definitions evolve (new lag/rolling columns) without an
    # ALTER TABLE — consumers select on this to know how to interpret
    # `feature_values`.
    Column("feature_set_version", String(32), nullable=False),
    Column("feature_values", JSON, nullable=False),
)
Index("ix_features_timestamp", features.c.timestamp)

# ---------------------------------------------------------------------------
# Model predictions (Forecasting: Phase 4, Contention: Phase 5)
# ---------------------------------------------------------------------------

forecast_predictions = Table(
    "forecast_predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    # When the prediction was generated.
    Column("generated_at", Float, nullable=False),
    # The future timestamp this prediction is for.
    Column("target_timestamp", Float, nullable=False),
    Column("metric_name", String(64), nullable=False),
    # "short" (30-60s) | "medium" (2-5min) — Section 10.2.
    Column("horizon_label", String(16), nullable=False),
    Column("horizon_seconds", Integer, nullable=False),
    Column("predicted_value", Float, nullable=False),
    # Model-derived confidence (FR-5.2) — not statistically calibrated in V1
    # (ADL-11).
    Column("confidence", Float, nullable=True),
    Column("model_version", String(64), nullable=False),
)
Index("ix_forecast_predictions_generated_at", forecast_predictions.c.generated_at)
Index("ix_forecast_predictions_target_timestamp", forecast_predictions.c.target_timestamp)

contention_predictions = Table(
    "contention_predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("generated_at", Float, nullable=False),
    Column("probability", Float, nullable=False),
    Column("threshold_used", Float, nullable=False),
    Column("triggered", Boolean, nullable=False),
    # [{"metric": "cpu_percent", "contribution": 0.62}, ...] — FR-6.2.
    Column("contributing_metrics", JSON, nullable=True),
    # [{"pid": 1234, "name": "chrome", "value": 0.41}, ...] — FR-6.2. "value"
    # is whichever raw_process_metrics column was ranked on to resolve the
    # dominant contributing metric (e.g. cpu_percent, memory_percent) — see
    # aiperfopt.contention.processes.get_contributing_processes(). Kept
    # distinct from contributing_metrics' "contribution" key deliberately:
    # that key is a signed classifier-coefficient contribution, this one is
    # a plain observed value, and conflating the two names here previously
    # made the two payload shapes look identical when they are not.
    Column("contributing_processes", JSON, nullable=True),
    # From Anomaly Detection, modulating confidence (FR-7.2). Nullable
    # because a contention prediction can occur without a concurrent
    # anomaly-detection run in the same cycle.
    Column("anomaly_score", Float, nullable=True),
    Column("model_version", String(64), nullable=False),
)
Index("ix_contention_predictions_generated_at", contention_predictions.c.generated_at)

# ---------------------------------------------------------------------------
# Evaluation results (Model Evaluation & Validation, Phase 4)
# ---------------------------------------------------------------------------

evaluations = Table(
    "evaluations",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("evaluated_at", Float, nullable=False),
    Column("model_name", String(64), nullable=False),
    Column("model_version", String(64), nullable=False),
    # Null for evaluations that aren't per-metric (reserved for future
    # cross-metric evaluation summaries).
    Column("metric_name", String(64), nullable=True),
    Column("horizon_label", String(16), nullable=True),
    # Walk-forward fold index (Section 11) — null for a summary row
    # averaged across folds.
    Column("fold_index", Integer, nullable=True),
    # e.g. "walk_forward", "holdout_scenario_1" (Section 11's mandatory
    # held-out scenario).
    Column("dataset_tag", String(64), nullable=False),
    Column("mae", Float, nullable=True),
    Column("rmse", Float, nullable=True),
    Column("directional_accuracy", Float, nullable=True),
    # Null when this row IS the primary model's result; set to e.g.
    # "naive_persistence" when this row is a baseline comparison result
    # (FR-10.2).
    Column("baseline_name", String(64), nullable=True),
    Column("extra", JSON, nullable=True),
)
Index("ix_evaluations_evaluated_at", evaluations.c.evaluated_at)

# ---------------------------------------------------------------------------
# Model deployment decisions (Model Retraining, Phase 8, FR-11.2/11.3, ADL-23)
# ---------------------------------------------------------------------------
#
# One row per human-confirmed deployment decision (or a system-initiated
# rollback). This is the "active version" audit trail: which candidate
# model_version was proposed, what it was compared against, what the
# comparison recommended, and what a human actually decided — never
# inferred from "whatever's newest on disk" (FR-11.2's "never silently
# activate"). ``aiperfopt.retraining.active_version`` reads/writes a
# small pointer file for *fast* "what's active right now" lookups on the
# live inference path; this table is the durable, queryable history
# behind that pointer, and is what NFR-5.3's automatic fallback walks
# backwards through to find the last known-good deployed version.

model_deployments = Table(
    "model_deployments",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("decided_at", Float, nullable=False),
    # The model_version a retraining run produced and proposed for
    # deployment (shared across forecasting/contention/anomaly artifacts
    # trained together in one retraining run — see
    # aiperfopt.retraining.pipeline's module docstring).
    Column("candidate_version", String(64), nullable=False),
    # The active version immediately before this decision, if any.
    Column("previous_version", String(64), nullable=True),
    # "deployed" | "rejected" | "rollback".
    Column("decision", String(16), nullable=False),
    # What aiperfopt.retraining.comparison recommended before a human
    # decided — "deploy" | "reject" | "no_active_baseline". Null for a
    # system-initiated rollback, which isn't comparison-driven.
    Column("recommendation", String(32), nullable=True),
    # Who/what made the decision — e.g. "human" once Phase 9's API wires
    # a real confirmation action, or "system" for an automatic rollback
    # (NFR-5.3).
    Column("confirmed_by", String(64), nullable=True),
    # Serialized aiperfopt.retraining.comparison.RetrainingComparison (or
    # a rollback reason) — kept for audit/debugging without needing to
    # re-derive the comparison from raw evaluations rows later.
    Column("comparison_summary", JSON, nullable=True),
)
Index("ix_model_deployments_decided_at", model_deployments.c.decided_at)

# ---------------------------------------------------------------------------
# Diagnosis results (Diagnosis / Explainable AI, Phase 6)
# ---------------------------------------------------------------------------
#
# One row per resolved diagnosis event (FR-8.3: every prediction event
# that warrants one resolves to exactly one of Section 12's eight
# taxonomy categories, ADL-24). Deliberately its own table rather than
# extra columns bolted onto contention_predictions: Diagnosis is a
# distinct subsystem (Section 7.1, item 8) that *consumes*
# Contention/Forecasting output (Section 7.3), and giving it its own
# table keeps that consumer relationship visible in the schema instead of
# blurring the two subsystems' boundaries the way ADL-19 already warns
# against for the Recommendation Engine.

diagnoses = Table(
    "diagnoses",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("generated_at", Float, nullable=False),
    # One of DiagnosisCategory's eight values (aiperfopt.diagnosis.taxonomy) —
    # stored as the enum's string value, not a foreign-key/int code, since
    # the taxonomy is a small, spec-fixed set (Section 12) rather than
    # user-editable reference data.
    Column("category", String(64), nullable=False),
    # False only for "unusual_pattern_anomaly_flagged" (ADL-36): that
    # category is informational only and must never reach the
    # Recommendation Engine (Phase 7 reads this column to enforce that,
    # rather than special-casing the category string itself).
    Column("is_actionable", Boolean, nullable=False),
    # Contention probability for the 7 actionable categories; anomaly
    # score for the anomaly-flagged category (Section 12's row uses a
    # different underlying signal for that one category — see
    # aiperfopt.diagnosis.service's module docstring).
    Column("confidence", Float, nullable=False),
    # "lightweight" or "shap" — which tier actually produced
    # forecast_explanations below (Section 10.5, NFR-1.3).
    Column("explanation_tier", String(16), nullable=False),
    # Why that tier fired: "high_confidence_contention",
    # "significant_anomaly", "explicit_request", or "routine" — kept
    # distinct from explanation_tier itself so "shap fired because of an
    # explicit request" stays distinguishable from "shap fired because
    # this was already a high-confidence contention prediction" even
    # though both produce tier="shap".
    Column("trigger_reason", String(32), nullable=False),
    Column("technical_explanation", Text, nullable=False),
    Column("plain_language_explanation", Text, nullable=False),
    # Per-metric SHAP/native-importance attribution behind
    # technical_explanation — [{"metric_name": ..., "forecaster_kind": ...,
    # "tier": ..., "top_features": [{"feature_name":..,"value":..}, ...],
    # "note": ...}, ...]. Retained even though only a short natural-
    # language summary is user-facing, per FR-8.3's "underlying
    # multi-resource contribution data is retained internally... exposed
    # via Explainable AI regardless of which single label is user-facing."
    Column("forecast_explanations", JSON, nullable=True),
    # Passed through unchanged from the triggering ContentionPredictionRecord
    # (FR-6.2) — Diagnosis reuses these rather than recomputing them, since
    # a logistic-regression coefficient contribution already *is* the exact
    # SHAP-linear attribution for that model (see
    # aiperfopt.diagnosis.explainer's module docstring for the numerical
    # proof this claim is checked against, not just asserted).
    Column("contributing_metrics", JSON, nullable=True),
    Column("contributing_processes", JSON, nullable=True),
    Column("anomaly_score", Float, nullable=True),
    Column("model_version", String(64), nullable=False),
)
Index("ix_diagnoses_generated_at", diagnoses.c.generated_at)

# ---------------------------------------------------------------------------
# Recommendations (Recommendation Engine, Phase 7)
# ---------------------------------------------------------------------------
#
# One row per generated recommendation set (FR-9.1: consumes only a
# standardized diagnosis category; FR-9.3: traceable back to the specific
# diagnosis that produced it). Deliberately rule-based, not a trained
# model (ADL-15) — there is no "model_version" column here the way
# forecast_predictions/contention_predictions/diagnoses have, because
# nothing here was fit to data; catalog_version instead identifies which
# revision of the curated category->actions mapping
# (aiperfopt/resources/recommendations.yaml) produced these actions, which is the
# actual thing that can change over time and that FR-9.3 needs traced.
#
# No SQL FOREIGN KEY constraint on diagnosis_id: nothing else in this
# schema uses one either (see ADL-17's SQLAlchemy Core choice) — kept
# consistent with the existing style rather than introducing the first
# FK constraint in this one table. diagnosis_category is stored
# alongside diagnosis_id in denormalized form deliberately, mirroring how
# contention_predictions already stores its own contributing_metrics/
# contributing_processes rather than requiring a join back to
# forecast_predictions — a Reporting or Frontend query filtering by
# category shouldn't need to join back to diagnoses just to do that.

recommendations = Table(
    "recommendations",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("generated_at", Float, nullable=False),
    Column("diagnosis_id", Integer, nullable=False),
    # Denormalized copy of diagnoses.category at generation time — see
    # the table-level comment above for why.
    Column("diagnosis_category", String(64), nullable=False),
    # [{"title": ..., "description": ..., "rationale": ...}, ...], ranked
    # highest-priority first — FR-9.1's actions curated by
    # aiperfopt.recommendation.catalog from aiperfopt/resources/recommendations.yaml.
    Column("actions", JSON, nullable=False),
    Column("catalog_version", String(64), nullable=False),
)
Index("ix_recommendations_generated_at", recommendations.c.generated_at)

# ---------------------------------------------------------------------------
# Structured logs (FR-3.1's "logs" category)
# ---------------------------------------------------------------------------
#
# This is distinct from the file-based JSON logging configured in
# aiperfopt.applog (Phase 0). That sink captures every log line for
# operational tailing/debugging (NFR-8.1) and would violate NFR-1.1's
# overhead budget if every line were also written to SQLite. This table
# is instead a queryable record of *significant* pipeline events
# (explicitly persisted by a subsystem via aiperfopt.storage.writer, not
# a logging.Handler mirroring every line) — e.g. a retraining outcome, a
# high-confidence contention prediction's supporting log entry, or an
# error a subsystem wants visible to the Reporting subsystem (Phase 11)
# without it having to parse log files.

app_logs = Table(
    "app_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", Float, nullable=False),
    Column("level", String(16), nullable=False),
    # Pipeline stage — conventionally the producing module's __name__,
    # mirroring the logger-name-as-stage-tag convention from Phase 0.
    Column("logger", String(128), nullable=False),
    Column("message", Text, nullable=False),
    Column("context", JSON, nullable=True),
)
Index("ix_app_logs_timestamp", app_logs.c.timestamp)
