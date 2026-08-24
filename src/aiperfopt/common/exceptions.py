"""
Shared exception hierarchy for the AI System Performance Optimizer.

Every subsystem raises exceptions from this module (or a subsystem-specific
subclass defined here) rather than letting arbitrary built-in exceptions
propagate. This gives calling code — particularly the Monitoring Engine's
per-source degradation logic (FR-1.4) and the API layer's error responses —
a single, predictable set of types to catch and handle.

Design notes
------------
- `AppError` is the root of the hierarchy. Nothing in this codebase should
  raise a bare `Exception`.
- Each subsystem in Section 7.1 of PROJECT_SPEC.md gets its own subclass so
  that error handling can be scoped precisely (e.g. "catch any monitoring
  error and continue in degraded mode" without accidentally swallowing a
  storage error that should propagate).
- `is_recoverable` lets a caller decide, generically, whether a failure is
  the kind that should trigger graceful degradation (NFR-5.1, NFR-5.4) or
  whether it represents a programming/configuration error that should stop
  the affected subsystem outright. Subsystems may override this per raise
  site by passing `recoverable=False` explicitly.
"""

from __future__ import annotations


class AppError(Exception):
    """Root of the application's exception hierarchy.

    Parameters
    ----------
    message:
        Human-readable description of what went wrong.
    recoverable:
        Whether the caller should treat this as a transient/degradable
        failure (log + skip / fall back) rather than a fatal one. Defaults
        to True, since most runtime failures in this system (a single
        metric source hiccupping, a storage write timing out) are expected
        to be handled gracefully rather than crashing the pipeline.
    context:
        Optional structured context (e.g. {"metric_source": "disk_io"})
        that logging call sites can attach to the structured log record
        without re-parsing the message string.
    """

    def __init__(
        self,
        message: str,
        *,
        recoverable: bool = True,
        context: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.recoverable = recoverable
        self.context = context or {}

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class ConfigError(AppError):
    """Configuration loading, validation, or override resolution failed."""


class LoggingSetupError(AppError):
    """Structured logging could not be initialized."""


class StorageError(AppError):
    """Data Storage subsystem failure (Section 7.1, item 3)."""


class StorageConnectionError(StorageError):
    """Could not establish or use a connection to the storage engine."""


class StorageSchemaError(StorageError):
    """Schema creation, migration, or validation failure."""


class MonitoringError(AppError):
    """Monitoring Engine failure (Section 7.1, item 1).

    Per FR-1.4, individual metric-source failures must not halt the whole
    engine. Raise sites for a single source's collection failure should use
    `recoverable=True` (the default) so the engine can log and skip.
    """


class MetricSourceUnavailableError(MonitoringError):
    """A specific metric source (e.g. IOKit power stats) is unavailable."""


class IngestionError(AppError):
    """Data Ingestion / Collector failure (Section 7.1, item 2)."""


class FeatureEngineeringError(AppError):
    """Feature Engineering failure (Section 7.1, item 4)."""


class ForecastingError(AppError):
    """Forecasting Models failure (Section 7.1, item 5)."""


class ContentionPredictionError(AppError):
    """Contention Prediction failure (Section 7.1, item 6)."""


class AnomalyDetectionError(AppError):
    """Anomaly Detection failure (Section 7.1, item 7)."""


class DiagnosisError(AppError):
    """Diagnosis / Explainable AI failure (Section 7.1, item 8)."""


class RecommendationError(AppError):
    """Recommendation Engine failure (Section 7.1, item 9)."""


class EvaluationError(AppError):
    """Model Evaluation & Validation failure (Section 7.1, item 10)."""


class RetrainingError(AppError):
    """Model Retraining failure (Section 7.1, item 11)."""


class ModelDeploymentError(RetrainingError):
    """A retrained model failed validation or deployment could not proceed.

    Deployment failures are never silently retried automatically beyond the
    documented fallback-to-last-known-good behavior (FR-11.3, NFR-5.3).
    """


class APIError(AppError):
    """API Layer failure (Section 7.1, item 12)."""


class FrontendError(AppError):
    """Frontend failure (Section 7.1, item 13)."""


class ReportingError(AppError):
    """Reporting / Analytics failure (Section 7.1, item 14)."""
