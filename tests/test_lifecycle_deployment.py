from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine

from aiperfopt.common.exceptions import ModelDeploymentError
from aiperfopt.config.settings import RetrainingSettings
from aiperfopt.lifecycle.comparison import RetrainingComparison
from aiperfopt.lifecycle.gate import SkillGateResult
from aiperfopt.lifecycle.deployment import (
    confirm_deployment,
    get_active_model_version,
    rollback_to_last_known_good,
)
from aiperfopt.storage import get_model_deployments_range

_PASSING_GATE = SkillGateResult(passed=True)
_FAILING_GATE = SkillGateResult(passed=False, reasons=["contention_classifier: F1 0.100 falls short of threshold-rule baseline 0.500 by more than the allowed 0.050"])


def _retraining_settings(tmp_path: Path) -> RetrainingSettings:
    return RetrainingSettings(
        active_version_file=tmp_path / "active_model_version.json",
        regression_tolerance=0.1,
        real_usage_holdout_fraction=0.2,
    )


@pytest.mark.plumbing
def test_get_active_model_version_none_before_any_deployment(storage_engine: Engine, tmp_path: Path):
    settings = _retraining_settings(tmp_path)
    assert get_active_model_version(settings) is None


@pytest.mark.behavioural
def test_confirm_deployment_true_activates_and_records(storage_engine: Engine, tmp_path: Path):
    settings = _retraining_settings(tmp_path)
    comparison = RetrainingComparison(candidate_version="v1", active_version=None, recommendation="no_active_baseline")

    record = confirm_deployment(storage_engine, settings, "v1", comparison, _PASSING_GATE, confirmed=True)

    assert record.decision == "deployed"
    assert record.candidate_version == "v1"
    assert record.previous_version is None
    assert get_active_model_version(settings) == "v1"

    stored = get_model_deployments_range(storage_engine)
    assert len(stored) == 1
    assert stored[0].decision == "deployed"


@pytest.mark.behavioural
def test_confirm_deployment_true_on_a_failing_gate_raises_and_does_not_activate(
    storage_engine: Engine, tmp_path: Path
):
    """Phase 6's core defect fix: human confirmation is necessary but no
    longer sufficient. Even ``confirmed=True`` must not activate a
    candidate that fails the quality gate -- and the refusal is still
    recorded to the audit trail (FR-11.2, R-8), not silently dropped."""
    settings = _retraining_settings(tmp_path)
    comparison = RetrainingComparison(candidate_version="v1", active_version=None, recommendation="no_active_baseline")

    with pytest.raises(ModelDeploymentError):
        confirm_deployment(storage_engine, settings, "v1", comparison, _FAILING_GATE, confirmed=True)

    assert get_active_model_version(settings) is None

    stored = get_model_deployments_range(storage_engine)
    assert len(stored) == 1
    assert stored[0].decision == "gate_failed"


@pytest.mark.behavioural
def test_confirm_deployment_false_does_not_activate(storage_engine: Engine, tmp_path: Path):
    settings = _retraining_settings(tmp_path)
    comparison = RetrainingComparison(candidate_version="v1", active_version=None, recommendation="deploy")

    record = confirm_deployment(storage_engine, settings, "v1", comparison, _PASSING_GATE, confirmed=False)

    assert record.decision == "rejected"
    assert get_active_model_version(settings) is None
    stored = get_model_deployments_range(storage_engine)
    assert len(stored) == 1
    assert stored[0].decision == "rejected"


@pytest.mark.behavioural
def test_confirm_deployment_records_previous_version(storage_engine: Engine, tmp_path: Path):
    settings = _retraining_settings(tmp_path)
    comparison1 = RetrainingComparison(candidate_version="v1", active_version=None, recommendation="no_active_baseline")
    confirm_deployment(storage_engine, settings, "v1", comparison1, _PASSING_GATE, confirmed=True)

    comparison2 = RetrainingComparison(candidate_version="v2", active_version="v1", recommendation="deploy")
    record2 = confirm_deployment(storage_engine, settings, "v2", comparison2, _PASSING_GATE, confirmed=True)

    assert record2.previous_version == "v1"
    assert get_active_model_version(settings) == "v2"


@pytest.mark.behavioural
def test_rollback_to_last_known_good_restores_previous_deployed_version(storage_engine: Engine, tmp_path: Path):
    settings = _retraining_settings(tmp_path)
    confirm_deployment(
        storage_engine, settings, "v1",
        RetrainingComparison(candidate_version="v1", active_version=None, recommendation="no_active_baseline"),
        _PASSING_GATE,
        confirmed=True,
    )
    confirm_deployment(
        storage_engine, settings, "v2",
        RetrainingComparison(candidate_version="v2", active_version="v1", recommendation="deploy"),
        _PASSING_GATE,
        confirmed=True,
    )
    assert get_active_model_version(settings) == "v2"

    record = rollback_to_last_known_good(storage_engine, settings, reason="v2 failed at inference time")

    assert record.decision == "rollback"
    assert record.candidate_version == "v1"
    assert record.previous_version == "v2"
    assert get_active_model_version(settings) == "v1"


@pytest.mark.behavioural
def test_rollback_skips_a_rejected_candidate(storage_engine: Engine, tmp_path: Path):
    settings = _retraining_settings(tmp_path)
    confirm_deployment(
        storage_engine, settings, "v1",
        RetrainingComparison(candidate_version="v1", active_version=None, recommendation="no_active_baseline"),
        _PASSING_GATE,
        confirmed=True,
    )
    confirm_deployment(
        storage_engine, settings, "v2",
        RetrainingComparison(candidate_version="v2", active_version="v1", recommendation="reject"),
        _PASSING_GATE,
        confirmed=False,  # rejected -- never became active
    )
    confirm_deployment(
        storage_engine, settings, "v3",
        RetrainingComparison(candidate_version="v3", active_version="v1", recommendation="deploy"),
        _PASSING_GATE,
        confirmed=True,
    )
    assert get_active_model_version(settings) == "v3"

    record = rollback_to_last_known_good(storage_engine, settings, reason="v3 failed")

    assert record.candidate_version == "v1"  # not the rejected v2


@pytest.mark.behavioural
def test_rollback_raises_when_no_prior_deployed_version_exists(storage_engine: Engine, tmp_path: Path):
    settings = _retraining_settings(tmp_path)
    confirm_deployment(
        storage_engine, settings, "v1",
        RetrainingComparison(candidate_version="v1", active_version=None, recommendation="no_active_baseline"),
        _PASSING_GATE,
        confirmed=True,
    )

    with pytest.raises(ModelDeploymentError):
        rollback_to_last_known_good(storage_engine, settings, reason="v1 failed, nothing before it")
