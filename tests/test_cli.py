"""Phase 0: the command-line entrypoint.

The project previously had no entrypoint at all -- no ``console_scripts``
and no ``python -m aiperfopt`` -- so reaching a working system required
pasting a Python snippet out of the README. These tests cover the
argument surface and the one command that must work on a completely
fresh install: ``doctor``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from aiperfopt.cli import build_parser, main
from aiperfopt.paths import HOME_ENV_VAR

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def clean_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key in list(os.environ):
        if key.startswith("AIPERFOPT_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(HOME_ENV_VAR, str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.mark.parametrize(
    "argv",
    [
        ["doctor"],
        ["collect", "--minutes", "1"],
        ["train", "--source", "real", "--version", "v2", "--deploy"],
        ["train", "--metrics", "cpu_percent", "mem_percent"],
        ["serve"],
        ["ui"],
        ["run"],
    ],
)
def test_every_subcommand_parses(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    assert callable(args.func)


def test_a_subcommand_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_train_defaults_are_the_documented_ones() -> None:
    args = build_parser().parse_args(["train"])
    assert args.source == "synthetic"
    assert args.version == "v1"
    assert args.metrics == ["cpu_percent", "mem_percent"]
    assert args.deploy is False


def test_doctor_succeeds_on_a_completely_fresh_install(clean_home: Path, capsys) -> None:
    """A fresh install is 'not ready yet', which is a normal state, not a failure."""
    assert main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "no_data" in out
    assert "aiperfopt collect" in out
    assert str(clean_home) in out


def test_console_script_is_declared_in_packaging_metadata() -> None:
    with (_PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    assert pyproject["project"]["scripts"]["aiperfopt"] == "aiperfopt.cli:main"


def test_python_dash_m_entrypoint_works(clean_home: Path) -> None:
    """``python -m aiperfopt`` must be equivalent to the installed script."""
    env = {**os.environ, HOME_ENV_VAR: str(clean_home)}
    result = subprocess.run(
        [sys.executable, "-m", "aiperfopt", "--help"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "aiperfopt" in result.stdout
