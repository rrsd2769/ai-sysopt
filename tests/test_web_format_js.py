"""Runs the dashboard's node tests and checks its labels cover the real taxonomy."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from aiperfopt.insight.taxonomy import DiagnosisCategory

JS_DIR = Path(__file__).resolve().parent.parent / "src" / "aiperfopt" / "web" / "static" / "js"

pytestmark = pytest.mark.behavioural


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_tests():
    files = sorted(str(p) for p in JS_DIR.glob("*.test.mjs"))
    assert len(files) >= 2
    result = subprocess.run(
        ["node", "--test", *files],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_diagnosis_category_has_a_web_label():
    source = (JS_DIR / "format.js").read_text()
    for category in DiagnosisCategory:
        assert f"  {category.value}: " in source, category.value
