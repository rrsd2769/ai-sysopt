from __future__ import annotations

from pathlib import Path

import pytest

from aiperfopt.common.exceptions import ConfigError
from aiperfopt.config import InterfaceMode, load_settings


def test_defaults_load_with_no_overrides(minimal_env):
    settings = load_settings()

    assert settings.mode.active_mode == InterfaceMode.GENERAL
    assert settings.monitoring.polling_interval_seconds == 1.5
    assert settings.forecasting.short_horizon_seconds == (30, 60)
    assert settings.forecasting.medium_horizon_seconds == (120, 300)
    assert settings.contention.decision_threshold == 0.75
    assert settings.privacy.excluded_processes == []


def test_user_override_file_beats_defaults(minimal_env, tmp_path: Path):
    override = tmp_path / "override.yaml"
    override.write_text(
        "monitoring:\n  polling_interval_seconds: 2.5\n"
        "privacy:\n  excluded_processes: ['spotlight']\n"
    )

    settings = load_settings(override)

    assert settings.monitoring.polling_interval_seconds == 2.5
    assert settings.privacy.excluded_processes == ["spotlight"]
    # Untouched fields still fall through to packaged defaults.
    assert settings.contention.decision_threshold == 0.75


def test_env_var_beats_user_override_file(minimal_env, tmp_path: Path, monkeypatch):
    override = tmp_path / "override.yaml"
    override.write_text("contention:\n  decision_threshold: 0.9\n")
    monkeypatch.setenv("AIPERFOPT_CONTENTION__DECISION_THRESHOLD", "0.42")

    settings = load_settings(override)

    assert settings.contention.decision_threshold == 0.42


def test_missing_user_override_file_falls_back_to_defaults(minimal_env, tmp_path: Path):
    nonexistent = tmp_path / "does_not_exist.yaml"

    settings = load_settings(nonexistent)

    assert settings.monitoring.polling_interval_seconds == 1.5


def test_invalid_horizon_range_rejected(minimal_env, tmp_path: Path):
    override = tmp_path / "bad.yaml"
    override.write_text("forecasting:\n  short_horizon_seconds: [60, 30]\n")

    with pytest.raises(ConfigError):
        load_settings(override)


def test_invalid_decision_threshold_rejected(minimal_env, tmp_path: Path):
    override = tmp_path / "bad.yaml"
    override.write_text("contention:\n  decision_threshold: 1.5\n")

    with pytest.raises(ConfigError):
        load_settings(override)


def test_malformed_yaml_raises_config_error(minimal_env, tmp_path: Path):
    override = tmp_path / "malformed.yaml"
    override.write_text("mode: [unclosed\n")

    with pytest.raises(ConfigError):
        load_settings(override)


def test_non_mapping_yaml_top_level_rejected(minimal_env, tmp_path: Path):
    override = tmp_path / "list_top_level.yaml"
    override.write_text("- just\n- a\n- list\n")

    with pytest.raises(ConfigError):
        load_settings(override)


def test_no_exclusions_by_default(minimal_env):
    """Section 13 / NFR-7.4: no process exclusions unless explicitly configured."""
    settings = load_settings()
    assert settings.privacy.excluded_processes == []


def test_developer_mode_selectable_via_env(minimal_env, monkeypatch):
    monkeypatch.setenv("AIPERFOPT_MODE__ACTIVE_MODE", "developer")
    settings = load_settings()
    assert settings.mode.active_mode == InterfaceMode.DEVELOPER
