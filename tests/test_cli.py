"""The command-line entrypoint.

The project previously had no entrypoint at all -- no ``console_scripts``
and no ``python -m aiperfopt`` -- so reaching a working system required
pasting a Python snippet out of the README. These tests cover the
argument surface and the one command that must work on a completely
fresh install: ``doctor``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
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


@pytest.mark.plumbing
@pytest.mark.parametrize(
    "argv",
    [
        ["doctor"],
        ["collect", "--minutes", "1"],
        ["collect", "--forever"],
        ["export-training-window", "--since", "0", "--out", "/tmp/export.json"],
        ["train", "--version", "v2", "--deploy"],
        ["train", "--metrics", "cpu_percent", "mem_percent"],
        ["evaluate", "--ablation"],
        ["evaluate", "--ablation", "--report"],
        ["evaluate", "--report", "--out-dir", "/tmp/results", "--metrics", "cpu_percent"],
        ["compare"],
        ["compare", "--raw", "--out-dir", "/tmp/results"],
        ["serve"],
        ["ui"],
        ["run"],
    ],
)
def test_every_subcommand_parses(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    assert callable(args.func)


@pytest.mark.plumbing
def test_a_subcommand_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


@pytest.mark.behavioural
def test_train_defaults_are_the_documented_ones() -> None:
    args = build_parser().parse_args(["train"])
    assert args.version == "v1"
    assert args.metrics == ["cpu_percent", "mem_percent"]
    assert args.deploy is False


@pytest.mark.behavioural
def test_evaluate_defaults_are_the_documented_ones() -> None:
    args = build_parser().parse_args(["evaluate"])
    assert args.ablation is False
    assert args.report is False
    assert args.metrics == ["cpu_percent", "mem_percent"]
    assert args.out_dir == Path("results")
    assert args.holdout_fraction == 0.2


@pytest.mark.plumbing
def test_evaluate_without_ablation_or_report_is_a_no_op_error(clean_home: Path) -> None:
    assert main(["evaluate"]) == 1


@pytest.mark.behavioural
def test_doctor_succeeds_on_a_completely_fresh_install(clean_home: Path, capsys) -> None:
    """A fresh install is 'not ready yet', which is a normal state, not a failure."""
    assert main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "no_data" in out
    assert "aiperfopt collect" in out
    assert str(clean_home) in out


@pytest.mark.plumbing
def test_console_script_is_declared_in_packaging_metadata() -> None:
    with (_PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    assert pyproject["project"]["scripts"]["aiperfopt"] == "aiperfopt.cli:main"


@pytest.mark.behavioural
def test_export_training_window_writes_system_metrics_and_process_summaries(
    clean_home: Path, tmp_path: Path
) -> None:
    from aiperfopt.bootstrap import bootstrap
    from aiperfopt.storage import (
        ProcessFeatureSummaryRecord,
        SystemMetricSample,
        insert_process_feature_summary,
        insert_system_metric_sample,
    )

    ctx = bootstrap()
    now = time.time()
    insert_system_metric_sample(
        ctx.storage_engine, SystemMetricSample(timestamp=now, cpu_percent=12.0, mem_percent=30.0)
    )
    insert_process_feature_summary(
        ctx.storage_engine,
        ProcessFeatureSummaryRecord(
            timestamp=now, computed_at=now, summary_values={"proc_cpu_sum": 5.0}
        ),
    )

    out_path = tmp_path / "export.json"
    assert main(["export-training-window", "--since", "0", "--out", str(out_path)]) == 0

    payload = json.loads(out_path.read_text())
    assert len(payload["system_metrics"]) == 1
    assert payload["system_metrics"][0]["cpu_percent"] == 12.0
    assert len(payload["process_feature_summaries"]) == 1
    assert payload["process_feature_summaries"][0]["proc_cpu_sum"] == 5.0


@pytest.mark.plumbing
def test_collect_forever_stops_cleanly_on_sigterm(clean_home: Path) -> None:
    """The background-collection mode: SIGTERM must flush and exit 0,
    not be killed mid-write the way the default SIGTERM handler would."""
    env = {**os.environ, HOME_ENV_VAR: str(clean_home)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "aiperfopt", "collect", "--forever"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        time.sleep(2.0)  # let the pipeline actually start and collect something
        proc.send_signal(signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=15)
        assert proc.returncode == 0, stderr
        assert "Stopped after" in stdout
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.mark.plumbing
def test_python_dash_m_entrypoint_works(clean_home: Path) -> None:
    """``python -m aiperfopt`` must be equivalent to the installed script."""
    env = {**os.environ, HOME_ENV_VAR: str(clean_home)}
    result = subprocess.run(
        [sys.executable, "-m", "aiperfopt", "--help"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "aiperfopt" in result.stdout
