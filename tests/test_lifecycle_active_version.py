from __future__ import annotations

from pathlib import Path

import pytest

from aiperfopt.common.exceptions import RetrainingError
from aiperfopt.lifecycle.active_version import read_active_version, write_active_version

pytestmark = pytest.mark.plumbing


def test_read_active_version_returns_none_when_file_absent(tmp_path: Path):
    assert read_active_version(tmp_path / "active_model_version.json") is None


def test_write_then_read_round_trips(tmp_path: Path):
    path = tmp_path / "nested" / "active_model_version.json"
    write_active_version(path, "v1")
    assert read_active_version(path) == "v1"


def test_write_active_version_overwrites(tmp_path: Path):
    path = tmp_path / "active_model_version.json"
    write_active_version(path, "v1")
    write_active_version(path, "v2")
    assert read_active_version(path) == "v2"


def test_write_active_version_is_atomic_no_tmp_file_left_behind(tmp_path: Path):
    path = tmp_path / "active_model_version.json"
    write_active_version(path, "v1")
    assert not (tmp_path / "active_model_version.json.tmp").exists()
    assert path.exists()


def test_read_active_version_raises_on_corrupted_file(tmp_path: Path):
    path = tmp_path / "active_model_version.json"
    path.write_text("not valid json {{{")

    with pytest.raises(RetrainingError):
        read_active_version(path)
