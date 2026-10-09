"""
The live-inference loop: the piece of the system that turns "Monitoring
is collecting raw samples" into "there is a current prediction/diagnosis/
recommendation to show," continuously, while the API server is running.

Nothing before Phase 9 runs on a cadence like this. Phases 3-8 are all
one-shot calls (compute one feature vector, train one model, diagnose one
event) — deliberately, so each phase could be built and tested in
isolation (NFR-14.1). This loop is where they're finally wired together
into the running system Section 6's flagship demo actually needs: predict
*before* the user notices a slowdown, not just answer a query about it.

Threading, not asyncio, for the loop itself
------------------------------------------------
Every step in the pipeline this loop drives — reading from SQLite,
running a LightGBM/statsmodels forecaster, scikit-learn inference, SHAP
attribution — is synchronous, CPU/IO-bound library code with no async
variant. Following the same pattern already established for Monitoring
(ADL-18: a dedicated thread, not asyncio, for exactly this kind of
work), this loop runs on its own thread; the FastAPI/WebSocket layer
(which *is* async) only ever reads the loop's latest published
:class:`LiveState` snapshot, never blocks waiting on it.

Graceful degradation is not an afterthought (NFR-11.1/11.2)
------------------------------------------------------------
Every stage after "read recent metrics" can legitimately have nothing to
report yet: not enough trailing history for a feature vector, no model
deployed yet (Phase 8 never confirmed one), a deployed model that fails
to load, or a cycle where the forecast simply didn't cross the
contention threshold. None of these are treated as errors — the loop
degrades to whatever's available and moves on to the next cycle, exactly
mirroring how Monitoring Engine's own per-source degradation (FR-1.4)
already treats a failing metric source as "skip it, don't crash."
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from sqlalchemy import Engine

from aiperfopt.anomaly.notifications import emit_anomaly_notification_if_significant
from aiperfopt.anomaly.persistence import load_anomaly_detector
from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import AppError, ModelDeploymentError
from aiperfopt.config.settings import ContentionSettings, DiagnosisSettings, ForecastingSettings, RetrainingSettings
from aiperfopt.contention.dataset import build_live_contention_input_row
from aiperfopt.contention.persistence import contention_artifact_exists, load_contention_classifier
from aiperfopt.contention.prediction import predict_contention
from aiperfopt.insight.diagnosis import diagnose_contention_event
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.features.storage_bridge import compute_and_store_live_feature_vector
from aiperfopt.lifecycle.deployment import resolve_deployed_forecasters
from aiperfopt.insight.catalog import RecommendationCatalog
from aiperfopt.insight.recommendations import generate_recommendations
from aiperfopt.lifecycle.deployment import get_active_model_version, rollback_to_last_known_good
from aiperfopt.storage.reader import get_recent_system_metrics
from aiperfopt.storage.records import (
    ContentionPredictionRecord,
    DiagnosisRecord,
    ForecastPredictionRecord,
    RecommendationRecord,
    SystemMetricSample,
)
from aiperfopt.storage.writer import insert_contention_prediction, insert_forecast_predictions_batch

logger = get_logger(__name__)


@dataclass
class LiveState:
    """One snapshot of the live pipeline (FR-12.1's WebSocket payload)."""

    timestamp: float
    latest_metrics: SystemMetricSample | None = None
    feature_vector: dict[str, float] | None = None
    active_model_version: str | None = None
    forecasts: list[ForecastPredictionRecord] | None = None
    contention: ContentionPredictionRecord | None = None
    diagnosis: DiagnosisRecord | None = None
    recommendation: RecommendationRecord | None = None
    anomaly_notification: bool = False
    error: str | None = None


@dataclass
class LiveInferenceLoop:
    """Ticks the live pipeline on a fixed interval, on its own thread."""

    storage_engine: Engine
    feature_config: FeatureEngineeringConfig
    forecasting_settings: ForecastingSettings
    contention_settings: ContentionSettings
    diagnosis_settings: DiagnosisSettings
    retraining_settings: RetrainingSettings
    catalog: RecommendationCatalog
    interval_seconds: float
    target_metrics: list[str] | None = None
    on_update: Callable[[LiveState], None] | None = None

    _thread: threading.Thread | None = field(default=None, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _latest_state: LiveState | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="live-inference", daemon=True)
        self._thread.start()
        logger.info("Live inference loop started", extra={"interval_seconds": self.interval_seconds})

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("Live inference loop stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest_state(self) -> LiveState | None:
        with self._lock:
            return self._latest_state

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                state = self._tick()
            except Exception as exc:  # noqa: BLE001 - a live loop must never die from one bad cycle
                logger.error("Live inference cycle failed", extra={"error": str(exc)})
                state = LiveState(timestamp=time.time(), error=str(exc))
            with self._lock:
                self._latest_state = state
            if self.on_update is not None:
                try:
                    self.on_update(state)
                except Exception:  # noqa: BLE001 - a broadcast failure must not kill the loop either
                    logger.error("Live state broadcast callback raised", exc_info=True)
            self._stop_event.wait(self.interval_seconds)

    def _tick(self) -> LiveState:
        state = LiveState(timestamp=time.time())

        recent = get_recent_system_metrics(
            self.storage_engine, window_seconds=self.feature_config.min_history_seconds
        )
        if recent:
            state.latest_metrics = recent[-1]

        feature_record = compute_and_store_live_feature_vector(self.storage_engine, self.feature_config)
        if feature_record is None:
            return state  # not enough trailing history yet — NFR-11.1
        state.feature_vector = feature_record.feature_values

        active_version = get_active_model_version(self.retraining_settings)
        state.active_model_version = active_version
        if active_version is None:
            return state  # nothing deployed yet (Phase 8 never confirmed one) — NFR-11.1

        forecasters, anomaly_detector, classifier = self._load_active_models(active_version, state)
        if forecasters is None or anomaly_detector is None:
            return state  # NFR-5.3's fallback already attempted inside _load_active_models

        row = build_live_contention_input_row(
            recent, feature_record, forecasters, anomaly_detector, self.feature_config
        )
        if row is None:
            return state  # a target metric has no usable current value this cycle

        forecast_records = self._build_forecast_records(
            row, feature_record.timestamp, list(forecasters.keys()), active_version
        )
        if forecast_records:
            insert_forecast_predictions_batch(self.storage_engine, forecast_records)
        state.forecasts = forecast_records

        anomaly_score = float(row["anomaly_score"])
        state.anomaly_notification = emit_anomaly_notification_if_significant(
            self.storage_engine, anomaly_score, self.contention_settings.anomaly_significant_threshold,
            feature_record.timestamp,
        )

        if classifier is None:
            return state  # Phase 6: this version's training data was single-class -- nothing to classify yet

        # Phase 5: a classifier trained since then carries its own
        # PR-curve-fitted decision threshold (aiperfopt.contention.
        # evaluation.fit_decision_threshold) -- prefer it over the static
        # config constant. An older artifact (or one never run through
        # the training pipeline's holdout step) has decision_threshold
        # unset (None), and falls back to the config value exactly as
        # before Phase 5.
        decision_threshold = (
            classifier.decision_threshold
            if classifier.decision_threshold is not None
            else self.contention_settings.decision_threshold
        )
        contention_record = predict_contention(
            classifier, self.storage_engine, row, as_of=feature_record.timestamp,
            decision_threshold=decision_threshold,
            anomaly_dampening_factor=self.contention_settings.anomaly_dampening_factor,
            contributing_processes_top_n=self.contention_settings.contributing_processes_top_n,
            model_version=active_version,
        )
        contention_record.id = insert_contention_prediction(self.storage_engine, contention_record)
        state.contention = contention_record

        diagnosis = diagnose_contention_event(
            self.storage_engine, contention_record, row, forecasters,
            self.contention_settings, self.diagnosis_settings, model_version=active_version,
        )
        state.diagnosis = diagnosis
        if diagnosis is None:
            return state  # nothing rose to the level of a diagnosis this cycle (Section 12)

        state.recommendation = generate_recommendations(self.storage_engine, diagnosis, self.catalog)
        return state

    def _build_forecast_records(
        self, row: Any, generated_at: float, metric_names: list[str], model_version: str
    ) -> list[ForecastPredictionRecord]:
        """One :class:`ForecastPredictionRecord` per deployed forecaster (Phase 7 / F5).

        ``row`` already carries each target's ``forecast_{metric}``/
        ``confidence_{metric}`` columns (:func:`build_live_contention_input_row`
        computed them once, for the classifier) — reused here rather than
        calling ``forecaster.predict()`` a second time. Horizon comes from
        settings, not the forecaster object: ``_load_active_models`` always
        resolves forecasters with ``horizon_label="short"``, and a deployed
        *baseline* forecaster (Section 10.7's contingency plan) carries no
        ``horizon_seconds``/``horizon_label`` attribute of its own.
        """
        horizon_label = "short"
        horizon_seconds = self.forecasting_settings.short_horizon_target_seconds
        records: list[ForecastPredictionRecord] = []
        for metric_name in metric_names:
            predicted_value = row.get(f"forecast_{metric_name}")
            if predicted_value is None:
                continue
            confidence = row.get(f"confidence_{metric_name}")
            if isinstance(confidence, float) and confidence != confidence:
                confidence = None  # NaN -> None; pandas coerces a per-metric None to NaN in a float Series
            records.append(
                ForecastPredictionRecord(
                    generated_at=generated_at,
                    target_timestamp=generated_at + horizon_seconds,
                    metric_name=metric_name,
                    horizon_label=horizon_label,
                    horizon_seconds=horizon_seconds,
                    predicted_value=float(predicted_value),
                    model_version=model_version,
                    confidence=confidence if confidence is None else float(confidence),
                )
            )
        return records

    def _load_active_models(
        self, active_version: str, state: LiveState
    ) -> tuple[dict[str, Any] | None, Any, Any | None]:
        """Load the forecasters/anomaly detector (required) and contention
        classifier (optional) for the active version.

        A *missing* contention classifier is not a load failure: Phase 6's
        quality gate deliberately allows deploying a version whose
        accumulated training data was still single-class for contention,
        in which case ``train_contention_and_anomaly_pipeline`` never
        trains or saves one at all (an expected, honest outcome — see
        Phase 5/6's handoffs). That must degrade to "no contention
        prediction/diagnosis/recommendation this cycle" (NFR-11.1), not
        trigger the same automatic-rollback path a genuinely
        missing/corrupted forecaster or anomaly-detector artifact does —
        the anomaly detector, unlike the classifier, is unconditionally
        trained and saved on every training run, so *its* absence does
        mean real corruption. Persisting forecasts (Phase 7 / F5) must not
        be held hostage to contention ever having trained a classifier.

        On any load failure of the required models (a missing/corrupted
        artifact — the concrete "deployment failure" NFR-5.3 has in
        mind), attempts one automatic rollback to the last known-good
        version and reports which happened via ``state.error`` — never
        raises out to the caller, since one bad model must not stop the
        loop (mirrors Monitoring's own per-source degradation, FR-1.4).
        """
        try:
            forecasters = resolve_deployed_forecasters(
                self.storage_engine, self.feature_config, self.forecasting_settings,
                active_version, horizon_label="short", target_metrics=self.target_metrics,
            )
            if not forecasters:
                state.error = f"model version {active_version!r} has no usable short-horizon forecasters"
                return None, None, None
            anomaly_detector = load_anomaly_detector(self.forecasting_settings.models_dir, active_version)
            classifier = (
                load_contention_classifier(self.forecasting_settings.models_dir, active_version)
                if contention_artifact_exists(self.forecasting_settings.models_dir, active_version)
                else None
            )
            return forecasters, anomaly_detector, classifier
        except (AppError, OSError, FileNotFoundError) as exc:
            logger.error(
                "Failed to load active model version, attempting rollback",
                extra={"model_version": active_version, "error": str(exc)},
            )
            try:
                rollback_record = rollback_to_last_known_good(
                    self.storage_engine, self.retraining_settings,
                    reason=f"inference-time load failure for {active_version}: {exc}",
                )
                state.error = (
                    f"model version {active_version!r} failed to load ({exc}); "
                    f"rolled back to {rollback_record.candidate_version!r}"
                )
            except ModelDeploymentError as rollback_exc:
                state.error = (
                    f"model version {active_version!r} failed to load ({exc}); "
                    f"no known-good version to roll back to ({rollback_exc})"
                )
            return None, None, None
