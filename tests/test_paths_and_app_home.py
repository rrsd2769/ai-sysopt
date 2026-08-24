"""Phase 0: the application data root, and every path derived from it.

These exist because runtime paths were previously resolved against the
*current working directory*: starting the server from one directory and
training from another silently produced two separate, empty databases
with no error anywhere. The behavioural claim under test is "the same
AIPERFOPT_HOME always reaches the same files, regardless of where the
process was launched from".
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aiperfopt.config import load_settings
from aiperfopt.paths import HOME_ENV_VAR, app_home, default_app_home, resolve_under_home


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key in list(os.environ):
        if key.startswith("AIPERFOPT_"):
            monkeypatch.delenv(key, raising=False)
    target = tmp_path / "apphome"
    monkeypatch.setenv(HOME_ENV_VAR, str(target))
    return target.resolve()


def test_app_home_honours_the_environment_override(home: Path) -> None:
    assert app_home() == home


def test_default_app_home_follows_windows_convention(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aiperfopt.paths.sys.platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\example\AppData\Local")
    assert default_app_home() == Path(r"C:\Users\example\AppData\Local") / "aiperfopt"


def test_default_app_home_falls_back_when_localappdata_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aiperfopt.paths.sys.platform", "win32")
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert default_app_home() == Path.home() / "AppData" / "Local" / "aiperfopt"


def test_default_app_home_follows_macos_convention(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aiperfopt.paths.sys.platform", "darwin")
    assert default_app_home() == Path.home() / "Library" / "Application Support" / "aiperfopt"


def test_relative_paths_resolve_under_home_not_cwd(home: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_under_home("data/x.db") == home / "data" / "x.db"


def test_absolute_paths_are_honoured_as_given(home: Path, tmp_path: Path) -> None:
    explicit = tmp_path / "elsewhere" / "custom.db"
    assert resolve_under_home(explicit) == explicit


def test_every_configured_path_is_absolute_and_under_home(home: Path) -> None:
    settings = load_settings()
    configured = [
        settings.storage.database_path,
        settings.forecasting.models_dir,
        settings.retraining.active_version_file,
        settings.logging.log_dir,
    ]
    for path in configured:
        assert path.is_absolute(), f"{path} is not absolute"
        assert home in path.parents or path == home, f"{path} does not live under {home}"


def test_same_home_reaches_same_database_from_different_working_directories(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression this whole module exists for."""
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    first_dir.mkdir()
    second_dir.mkdir()

    monkeypatch.chdir(first_dir)
    from_first = load_settings().storage.database_path

    monkeypatch.chdir(second_dir)
    from_second = load_settings().storage.database_path

    assert from_first == from_second
    # And nothing was created beside either working directory.
    assert list(first_dir.iterdir()) == []
    assert list(second_dir.iterdir()) == []
