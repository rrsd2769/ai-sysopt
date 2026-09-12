"""Phase 2 of docs/plans/2026-09-12-presentable-predictive-monitor.md:
the monitor must not be a burden on a machine it's expected to run on
passively for days. These are the tests that fail if a future change
makes the pipeline itself heavy.

The measurement itself lives in :mod:`aiperfopt.monitoring.overhead`
(extracted in Phase 5 so the same function also produces
``results/overhead.txt`` for the Review 2 evidence pack, instead of this
budget-enforcement logic being test-only) -- these tests just call it and
assert against its documented budgets.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperfopt.monitoring.overhead import (
    CPU_BUDGET_PERCENT,
    DB_GROWTH_CEILING_BYTES_PER_HOUR,
    measure_pipeline_overhead,
)


@pytest.mark.timeout(60)
def test_pipeline_self_cpu_stays_within_budget(tmp_path: Path):
    measurement = measure_pipeline_overhead(tmp_path)
    assert measurement.cpu_percent < CPU_BUDGET_PERCENT, (
        f"pipeline used {measurement.cpu_percent:.2f}% CPU, budget is {CPU_BUDGET_PERCENT}%"
    )


@pytest.mark.timeout(60)
def test_database_growth_stays_within_a_generous_ceiling(tmp_path: Path):
    measurement = measure_pipeline_overhead(tmp_path)
    assert measurement.db_growth_bytes_per_hour < DB_GROWTH_CEILING_BYTES_PER_HOUR, (
        f"database grew at an extrapolated "
        f"{measurement.db_growth_bytes_per_hour / (1024 * 1024):.1f} MB/hour, "
        f"ceiling is {DB_GROWTH_CEILING_BYTES_PER_HOUR / (1024 * 1024):.0f} MB/hour"
    )
