"""
Layered configuration for the AI System Performance Optimizer.

Implements FR-15.1 (layered configuration: defaults + user override, for
polling interval, contention threshold, forecast horizons, mode selection,
and process exclusions) and NFR-12.1 (all operational parameters externally
configurable without modifying application source code).

Priority order, highest to lowest:

    1. Environment variables      (prefix ``AIPERFOPT_``, nested delimiter ``__``)
    2. ``.env`` file               (if present)
    3. User override YAML file     (``AIPERFOPT_CONFIG_FILE`` or explicit path)
    4. Packaged default YAML file  (``aiperfopt/resources/default.yaml``)

This is implemented via pydantic-settings' ``settings_customise_sources``
hook rather than hand-merging dicts, so each layer's precedence is
enforced by the library itself and is easy to reason about / test in
isolation.

Path fields
-----------
Every ``Path`` field here is made absolute at validation time against
:func:`aiperfopt.paths.app_home`. Relative values in any layer are
therefore never interpreted against the current working directory --
which is what previously caused two launches from two directories to
reach two different, empty databases.
"""

from __future__ import annotations

import contextvars
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from aiperfopt.common.exceptions import ConfigError
from aiperfopt.paths import resolve_under_home
from aiperfopt.resources import DEFAULT_CONFIG_FILENAME, resource_path

_DEFAULT_CONFIG_PATH = resource_path(DEFAULT_CONFIG_FILENAME)

# Carries the user-override YAML path into settings_customise_sources at
# instantiation time. A ContextVar (rather than a mutable class attribute)
# is used deliberately: pydantic's metaclass intercepts underscore-prefixed
# *annotated* class attributes as private-attribute descriptors, which
# defeats a simple subclass-attribute override, and a ContextVar is also
# safely isolated across concurrent loads on different threads/tasks.
_user_config_path_var: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "aiperfopt_user_config_path", default=None
)


def _absolute_under_app_home(value: Path) -> Path:
    """Field validator: interpret a configured path against the app home.

    Applied to every path field so that a relative value in
    ``default.yaml`` can never be resolved against the current working
    directory. See :mod:`aiperfopt.paths`.
    """
    return resolve_under_home(value)


class InterfaceMode(str, Enum):
    """The two Version 1 interface modes (Section 4)."""

    GENERAL = "general"
    DEVELOPER = "developer"


class ModeSettings(BaseModel):
    active_mode: InterfaceMode = InterfaceMode.GENERAL


class MonitoringSettings(BaseModel):
    polling_interval_seconds: float = Field(gt=0)
    process_polling_interval_seconds: float = Field(gt=0)
    # Phase 2's storage bound: only the top-N processes by CPU, unioned
    # with the top-N by memory RSS, are individually stored per poll --
    # everything else is folded into one aggregate row (see
    # aiperfopt.monitoring.process_tracker). Launch/terminate events are
    # unaffected by this and are always recorded for every process.
    process_capture_top_n: int = Field(gt=0)

    @field_validator("polling_interval_seconds", "process_polling_interval_seconds")
    @classmethod
    def _reasonable_interval(cls, value: float) -> float:
        if value > 300:
            raise ValueError("polling interval > 300s is almost certainly a misconfiguration")
        return value


class ForecastingSettings(BaseModel):
    short_horizon_seconds: tuple[int, int]
    medium_horizon_seconds: tuple[int, int]
    # Number of chronological folds for walk-forward validation (Section 11,
    # ADL-10). sklearn's TimeSeriesSplit is the chronological-fold
    # primitive; this is its n_splits.
    walk_forward_splits: int = Field(gt=1)
    # A metric/horizon combination with fewer training rows than this
    # after target-alignment is not trained — LightGBM on a handful of
    # rows would overfit to noise rather than produce a meaningful model.
    min_training_rows: int = Field(gt=0)
    # Trained model artifacts (LightGBM boosters + metadata sidecars) are
    # written here, one subdirectory per metric/horizon.
    models_dir: Path
    # A candidate metric/horizon change is only counted as "stable" (not
    # rising/falling) if the relative change is below this fraction —
    # used for directional-accuracy scoring (Section 11: "correctly
    # predicting whether a metric is rising, falling, or stable").
    directional_stability_threshold: float = Field(gt=0, lt=1)
    # Section 10.7's contingency plan: LightGBM must beat every baseline's
    # mean MAE by at least this fraction to be selected over the best
    # baseline, preventing deployment flip-flopping on noise-level
    # differences.
    baseline_improvement_margin: float = Field(ge=0, lt=1)

    _abs_models_dir = field_validator("models_dir")(_absolute_under_app_home)

    @field_validator("short_horizon_seconds", "medium_horizon_seconds")
    @classmethod
    def _ordered_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        lo, hi = value
        if lo <= 0 or hi <= 0:
            raise ValueError("horizon bounds must be positive seconds")
        if lo >= hi:
            raise ValueError("horizon range must be [min, max] with min < max")
        return value

    @property
    def short_horizon_target_seconds(self) -> int:
        """The concrete regression target within the short-horizon range.

        The spec defines horizons as ranges (30-60s), not single points.
        Forecasting needs one concrete number of seconds ahead to train
        against; the range's midpoint is used as that representative
        point (documented design choice — see ADL-06's rationale for why
        this is one model, not one per exact second).
        """
        lo, hi = self.short_horizon_seconds
        return (lo + hi) // 2

    @property
    def medium_horizon_target_seconds(self) -> int:
        lo, hi = self.medium_horizon_seconds
        return (lo + hi) // 2


class ContentionSettings(BaseModel):
    # Conservative threshold favoring precision over recall (FR-6.3).
    decision_threshold: float = Field(ge=0.0, le=1.0)
    # How strongly Anomaly Detection's score reduces the reported
    # contention probability (FR-7.2: anomaly output "modulates
    # contention-prediction confidence"). 0 = no effect; 1 = a maximally
    # anomalous reading can zero out the reported probability entirely.
    anomaly_dampening_factor: float = Field(ge=0.0, le=1.0)
    # Anomaly score above which a distinct informational notification is
    # surfaced (FR-7.2), independent of any contention prediction.
    anomaly_significant_threshold: float = Field(ge=0.0, le=1.0)
    # How many top resource-consuming processes to report as "contributing
    # processes" (FR-6.2) once the dominant contributing metric is known.
    contributing_processes_top_n: int = Field(gt=0)


class DiagnosisSettings(BaseModel):
    """Rule parameters for taxonomy resolution (Section 12, ADL-24).

    These govern the *rule-based refinement* Diagnosis applies on top of
    Contention Prediction's already-learned output — e.g. "is this
    CPU-dominant event one runaway process or several" — not a trained
    model of their own, consistent with Section 3's preference for the
    simplest mechanism that does the job (the taxonomy itself is a fixed,
    spec-defined set, not something worth training a classifier for).
    Reuses ``ContentionSettings.anomaly_significant_threshold`` for the
    "Unusual Pattern (Anomaly-Flagged)" trigger rather than duplicating
    that threshold here — see ``aiperfopt.diagnosis.taxonomy``'s module
    docstring for why a second copy of that number would be a real
    config-drift risk.
    """

    # Below this fraction, no single resource type's share of total
    # contribution is considered "dominant" -> Cross-Resource Contention.
    cross_resource_dominance_ratio: float = Field(ge=0.0, le=1.0)
    # Within a CPU-dominant event, the top contributing process's share
    # of the top-N processes' combined activity must reach this fraction
    # to be labeled CPU-Bound Single Process rather than Multi-Process.
    process_dominance_ratio: float = Field(ge=0.0, le=1.0)
    # Absolute mem_percent_rate_of_change (percentage-points/second) at
    # or above which a memory-dominant event is labeled "Sudden" rather
    # than the default "Gradual".
    memory_sudden_rate_of_change_threshold: float = Field(gt=0.0)
    # How many per-feature SHAP/native-importance attributions to retain
    # per forecast explanation (FR-8.1).
    shap_top_n: int = Field(gt=0)


class RetrainingSettings(BaseModel):
    """Model Retraining tunables (Section 7.1 item 11, FR-11.1-11.3, ADL-23).

    Retraining reuses Phase 4/5's existing training pipelines verbatim
    (see ``aiperfopt.retraining.pipeline``'s module docstring) — nothing
    here configures *how* a model is trained, only how a freshly-trained
    candidate is compared against whatever is currently deployed, and
    where the "which version is currently active" pointer lives.
    """

    # Where the small active-model-version pointer file lives
    # (``aiperfopt.retraining.active_version``). Deliberately a plain
    # file, not a database row: the live inference path (Phase 9) needs
    # to answer "what's active right now" on every cycle, and a file
    # read is simpler and faster than a query for that single fact. The
    # durable, queryable *history* of deployment decisions still lives in
    # the ``model_deployments`` storage table (FR-11.2's audit trail).
    active_version_file: Path
    # A candidate's holdout metric is only considered "regressed" versus
    # the currently active version if it's worse by more than this
    # fraction — prevents blocking deployment on noise-level differences
    # (the same rationale as ForecastingSettings.baseline_improvement_margin,
    # applied to candidate-vs-active rather than model-vs-baseline).
    regression_tolerance: float = Field(ge=0.0, lt=1.0)
    # Fraction of chronologically-ordered accumulated real-usage data
    # reserved as a held-out generalization check: the most recent slice,
    # never an earlier one, to avoid future-to-past leakage.
    real_usage_holdout_fraction: float = Field(gt=0.0, lt=1.0)
    # Phase 6's quality gate (Section 11's precision-favoring stance,
    # extended from "don't regress vs. the active version" to "clear the
    # absolute bar every deployed model must meet"): a candidate forecaster
    # must beat the *persistence* baseline's holdout MAE by at least this
    # relative fraction. Deliberately separate from
    # ``ForecastingSettings.baseline_improvement_margin`` (which governs
    # LightGBM-vs-baseline model *selection*) -- this margin instead gates
    # *deployment* of whichever model was selected, baseline or not.
    forecasting_mae_margin: float = Field(default=0.05, ge=0.0, lt=1.0)
    # A candidate forecaster's holdout directional accuracy must exceed
    # 0.5 (chance) by at least this much to pass the gate.
    directional_accuracy_margin: float = Field(default=0.05, ge=0.0, lt=0.5)
    # A candidate contention classifier's holdout F1 (both measured on
    # the same holdout, see aiperfopt.contention.training_pipeline) may
    # trail the threshold-rule baseline's F1 by at most this many
    # absolute F1 points and still pass the gate -- a tolerance, not a
    # required margin of superiority, since F1's [0, 1] ceiling means an
    # already-excellent (even perfect) baseline on an easy holdout can be
    # legitimately impossible to "beat by a margin" (see
    # aiperfopt.retraining.comparison.evaluate_skill_gate).
    contention_f1_margin: float = Field(default=0.05, ge=0.0, lt=1.0)

    _abs_active_version_file = field_validator("active_version_file")(_absolute_under_app_home)


class ApiSettings(BaseModel):
    """API Layer tunables (Section 7.1 item 12, FR-12.1-12.4).

    Not among FR-15.1's explicitly named fields, but NFR-12.1 requires
    all operational parameters to be externally configurable, so the
    server bind address/port and the live-inference cadence get the same
    layered-config treatment as everything else rather than being
    hardcoded in ``api/app.py`` or a CLI flag default.
    """

    host: str = "127.0.0.1"
    port: int = Field(gt=0, le=65535)
    # How often the live-inference loop (feature -> forecast -> contention
    # -> diagnosis -> recommendation) ticks. Distinct from
    # MonitoringSettings.polling_interval_seconds, which controls raw
    # sample collection — live inference reads whatever's accumulated in
    # storage since its last tick, so it can run less frequently than raw
    # polling without losing samples (Ingestion, Phase 2, already
    # guarantees no sample is dropped on the write side).
    live_inference_interval_seconds: float = Field(gt=0)


class PrivacySettings(BaseModel):
    # Section 13 / FR-9.7 / NFR-7.4: user-controlled process exclusions,
    # empty by default.
    excluded_processes: list[str] = Field(default_factory=list)


class StorageSettings(BaseModel):
    database_path: Path
    full_resolution_retention_hours: int = Field(gt=0)
    # Independent, much shorter retention for process-level rows (Phase 2,
    # NFR-4.3) -- see aiperfopt.storage.rollup.apply_retention_policy for
    # why this is split from full_resolution_retention_hours rather than
    # sharing one cutoff.
    process_retention_hours: float = Field(gt=0)
    # Phase 9 F9: explicit, operator-set flag (AIPERFOPT_STORAGE__IS_REPLAY)
    # for pointing a running API server at a replay database for a demo.
    # Not auto-detected from database_path's filename -- `aiperfopt replay
    # --replay-db` accepts an arbitrary path, so sniffing the name would
    # silently stop working the moment someone uses a custom one. This is
    # the one signal readiness.assess_readiness has for "am I serving
    # replayed data right now," so the UI can show an honest REPLAY badge
    # instead of an indistinguishable "Live".
    is_replay: bool = False

    _abs_database_path = field_validator("database_path")(_absolute_under_app_home)


class IngestionSettings(BaseModel):
    """Tunables for the Data Ingestion consumer (Phase 2).
    contention threshold, forecast horizons, mode, process exclusions),
    but NFR-12.1 requires *all* operational parameters to be externally
    configurable, so these get the same layered-config treatment rather
    than being hardcoded constants in ``ingestion/consumer.py``.
    """

    # Max samples of a single record type accumulated before a batch is
    # flushed to storage, even if flush_interval_seconds hasn't elapsed.
    batch_size: int = Field(gt=0)
    # Max time a partial batch sits unflushed — bounds staleness of data
    # visible to the API/reporting layers when the sample rate is low.
    flush_interval_seconds: float = Field(gt=0)
    # Bound on the shared monitoring->ingestion queue (FR-2.2: Monitoring
    # must never block on Ingestion; a bounded queue with non-blocking
    # producer puts is how that's enforced under sustained backpressure).
    queue_maxsize: int = Field(gt=0)
    # How often the retention/rollup policy (FR-3.2) is applied.
    retention_check_interval_seconds: float = Field(gt=0)


class LoggingSettings(BaseModel):
    level: str = "INFO"
    log_dir: Path = Path("logs")
    json_format: bool = True

    _abs_log_dir = field_validator("log_dir")(_absolute_under_app_home)

    @field_validator("level")
    @classmethod
    def _valid_level(cls, value: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in valid:
            raise ValueError(f"logging level must be one of {sorted(valid)}, got {value!r}")
        return upper


class _YamlLayerSource(PydanticBaseSettingsSource):
    """A pydantic-settings source that reads a single YAML file.

    Missing files resolve to an empty layer rather than an error, since the
    user-override layer is optional by design (FR-15.1 only requires that
    overrides be *possible*, not present).
    """

    def __init__(self, settings_cls: type[BaseSettings], yaml_path: Path | None) -> None:
        super().__init__(settings_cls)
        self._yaml_path = yaml_path

    def get_field_value(self, field, field_name):  # pragma: no cover - unused path
        # Required by the abstract base but not used since we override
        # __call__ to return the whole parsed document at once.
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        if self._yaml_path is None or not self._yaml_path.is_file():
            return {}
        try:
            with self._yaml_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(
                f"Failed to load configuration file: {self._yaml_path}",
                recoverable=False,
                context={"path": str(self._yaml_path)},
            ) from exc
        if not isinstance(data, dict):
            raise ConfigError(
                f"Configuration file must contain a YAML mapping at the top level: "
                f"{self._yaml_path}",
                recoverable=False,
                context={"path": str(self._yaml_path)},
            )
        return data


class AppSettings(BaseSettings):
    """Root configuration object. Construct via :func:`load_settings`.

    Direct instantiation (``AppSettings()``) also works and follows the
    same layered precedence, since the layering is implemented in
    ``settings_customise_sources`` rather than in the loader function.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERFOPT_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    mode: ModeSettings = Field(default_factory=ModeSettings)
    monitoring: MonitoringSettings
    forecasting: ForecastingSettings
    contention: ContentionSettings
    diagnosis: DiagnosisSettings
    retraining: RetrainingSettings
    api: ApiSettings
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    storage: StorageSettings
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    ingestion: IngestionSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Order = precedence, highest first.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlLayerSource(settings_cls, _user_config_path_var.get()),
            _YamlLayerSource(settings_cls, _DEFAULT_CONFIG_PATH),
            file_secret_settings,
        )


def load_settings(user_config_path: Path | str | None = None) -> AppSettings:
    """Build the layered :class:`AppSettings` object.

    Parameters
    ----------
    user_config_path:
        Optional path to a user override YAML file (layer 3). If omitted,
        only environment variables and the packaged defaults apply.

    Raises
    ------
    ConfigError
        If a configured file exists but cannot be parsed, or if the merged
        configuration fails schema validation.
    """
    resolved = Path(user_config_path).expanduser() if user_config_path else None

    token = _user_config_path_var.set(resolved)
    try:
        return AppSettings()
    except ConfigError:
        raise
    except Exception as exc:  # pydantic ValidationError and friends
        raise ConfigError(
            "Configuration validation failed", recoverable=False, context={"error": str(exc)}
        ) from exc
    finally:
        _user_config_path_var.reset(token)
