"""Packaged configuration resources.

These YAML files live *inside* the package rather than in a top-level
``config/`` directory so they are present in a built wheel and in a
PyInstaller bundle. They previously resolved via
``Path(__file__).resolve().parents[3] / "config"``, which only ever
worked for an editable install: a real ``pip install .`` produced a
wheel containing no YAML at all, and every entrypoint died at
``load_settings()``.

Read them through :func:`resource_path`, never by reconstructing a path
relative to ``__file__`` in a calling module.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

DEFAULT_CONFIG_FILENAME = "default.yaml"
RECOMMENDATION_CATALOG_FILENAME = "recommendations.yaml"


def resource_path(filename: str) -> Path:
    """Absolute filesystem path to one packaged resource.

    ``importlib.resources.files`` returns a ``Traversable``; for a normal
    filesystem install (including PyInstaller's extracted bundle) that is
    already a real path, which is what the YAML loaders here want.
    """
    return Path(str(resources.files(__package__).joinpath(filename)))
