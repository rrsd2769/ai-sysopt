"""The YAML configuration must ship inside the distribution.

``config/default.yaml`` and ``config/recommendations.yaml`` used to live
in a top-level ``config/`` directory resolved via
``Path(__file__).resolve().parents[3]``. That only ever worked for an
editable install -- a built wheel contained no YAML at all (verified:
101 files, zero ``.yaml``), so ``pip install .`` and any PyInstaller
bundle died at ``load_settings()``. These tests assert the runtime
mechanism that replaced it.
"""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path

import pytest

from aiperfopt.insight.catalog import load_recommendation_catalog
from aiperfopt.resources import (
    DEFAULT_CONFIG_FILENAME,
    RECOMMENDATION_CATALOG_FILENAME,
    resource_path,
)

pytestmark = pytest.mark.plumbing

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_both_yaml_resources_are_importable_package_data() -> None:
    names = {entry.name for entry in resources.files("aiperfopt.resources").iterdir()}
    assert DEFAULT_CONFIG_FILENAME in names
    assert RECOMMENDATION_CATALOG_FILENAME in names


def test_resource_path_points_at_a_real_readable_file() -> None:
    for filename in (DEFAULT_CONFIG_FILENAME, RECOMMENDATION_CATALOG_FILENAME):
        path = resource_path(filename)
        assert path.is_file(), f"{filename} is not on disk at {path}"
        assert path.read_text(encoding="utf-8").strip(), f"{filename} is empty"


def test_packaging_metadata_declares_the_yaml_as_package_data() -> None:
    """Guards the packaging half: importable in a source tree is not enough."""
    with (_PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    assert "resources/*.yaml" in package_data["aiperfopt"]


def test_no_yaml_is_left_behind_in_a_top_level_config_directory() -> None:
    """A second copy would drift silently from the packaged one."""
    stale = list((_PROJECT_ROOT / "config").glob("*.yaml")) if (_PROJECT_ROOT / "config").is_dir() else []
    assert stale == [], f"unpackaged YAML still present: {stale}"


def test_catalog_loads_through_the_packaged_resource() -> None:
    catalog = load_recommendation_catalog()
    assert catalog.catalog_version
