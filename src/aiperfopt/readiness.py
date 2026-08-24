"""Answers one question: can this system currently produce a prediction, and if not, why?

Before this existed, a fresh install produced a live loop that returned
early on every cycle forever (no trailing history, then no deployed
model) while the UI displayed a green "Live" banner. The failure was
indistinguishable from success. Readiness makes each early-return
reason nameable, so both the CLI (``aiperfopt doctor``) and the UI can
say what is actually missing instead of showing a default.

Kept out of the API layer deliberately: ``aiperfopt doctor`` must be able
to answer this with no server running.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import Engine

from aiperfopt.config.settings import RetrainingSettings
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.retraining.deployment import get_active_model_version
from aiperfopt.storage.reader import get_recent_system_metrics


# How stale the newest sample may be before monitoring is treated as not
# running. Comfortably above any sane polling interval, low enough that a
# stopped collector is noticed within a cycle or two of looking.
STALE_AFTER_SECONDS = 60.0


class ReadinessState(str, Enum):
    NO_DATA = "no_data"
    COLLECTING = "collecting"
    NEEDS_TRAINING = "needs_training"
    READY = "ready"


@dataclass(frozen=True)
class Readiness:
    state: ReadinessState
    detail: str
    history_seconds: float
    history_seconds_required: float
    active_model_version: str | None

    @property
    def can_predict(self) -> bool:
        return self.state is ReadinessState.READY


def assess_readiness(
    engine: Engine,
    feature_config: FeatureEngineeringConfig,
    retraining_settings: RetrainingSettings,
) -> Readiness:
    """Report whether the live pipeline can produce a prediction right now.

    The two checks mirror, in order, the two early returns in
    :meth:`aiperfopt.api.live_state.LiveInferenceLoop._tick` -- not enough
    trailing history, then no deployed model. Keeping the order identical
    is what makes the reported reason the *actual* reason the loop would
    stop.
    """
    required = float(feature_config.min_history_seconds)
    active_version = get_active_model_version(retraining_settings)

    # Queried over a wider window than `required` deliberately: asking for
    # exactly `required` seconds caps the measurable span at the width of
    # the query itself, which would make READY unreachable. What matters
    # is the *coverage* the window actually contains.
    recent = get_recent_system_metrics(engine, window_seconds=required * 2)
    if not recent:
        return Readiness(
            state=ReadinessState.NO_DATA,
            detail="No metrics collected yet. Start monitoring with 'aiperfopt collect' or 'aiperfopt run'.",
            history_seconds=0.0,
            history_seconds_required=required,
            active_model_version=active_version,
        )

    span = max(0.0, recent[-1].timestamp - recent[0].timestamp)
    staleness = max(0.0, time.time() - recent[-1].timestamp)

    if staleness > STALE_AFTER_SECONDS:
        return Readiness(
            state=ReadinessState.COLLECTING,
            detail=(
                f"Monitoring does not appear to be running - the newest sample is "
                f"{staleness:.0f}s old. Start it with 'aiperfopt collect' or 'aiperfopt run'."
            ),
            history_seconds=span,
            history_seconds_required=required,
            active_model_version=active_version,
        )

    if span < required:
        return Readiness(
            state=ReadinessState.COLLECTING,
            detail=(
                f"Collecting data - {span:.0f}s of {required:.0f}s needed before "
                f"a full feature vector can be computed."
            ),
            history_seconds=span,
            history_seconds_required=required,
            active_model_version=active_version,
        )

    if active_version is None:
        return Readiness(
            state=ReadinessState.NEEDS_TRAINING,
            detail="Enough data, but no model is deployed. Run 'aiperfopt train --deploy'.",
            history_seconds=span,
            history_seconds_required=required,
            active_model_version=None,
        )

    return Readiness(
        state=ReadinessState.READY,
        detail=f"Ready - serving predictions from model version {active_version!r}.",
        history_seconds=span,
        history_seconds_required=required,
        active_model_version=active_version,
    )
