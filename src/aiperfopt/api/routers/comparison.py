"""
Model-comparison results for the dashboard's Compare view.

``aiperfopt compare`` (:mod:`aiperfopt.evaluation.model_comparison`) is
the only thing that *runs* the LightGBM / Random Forest / Ridge
comparison -- it takes minutes and rewrites the result files the paper
quotes, so the API never starts it. This router only reads what that run
wrote:

- ``model_comparison.csv`` -- holdout score per (target, model)
- ``model_comparison_cv.csv`` -- walk-forward score per (target, fold, model)
- ``paper/regimes.csv`` -- holdout MAE split into stable vs moving load,
  written by ``scripts/paper_analysis.py`` (optional)

The results directory is ``results/`` under the working directory, the
same default ``aiperfopt compare --out-dir`` uses; ``app.state.results_dir``
overrides it.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from aiperfopt.api.dependencies import get_app_context
from aiperfopt.bootstrap import AppContext

router = APIRouter(prefix="/comparison", tags=["comparison"])

DEFAULT_RESULTS_DIR = Path("results")
_NUMERIC = ("mae", "rmse", "directional_accuracy", "fit_seconds", "latency_ms", "size_kb", "share", "cut_pp")


def get_results_dir(request: Request) -> Path:
    return Path(getattr(request.app.state, "results_dir", DEFAULT_RESULTS_DIR))


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for key in _NUMERIC:
            if key in row:
                row[key] = float(row[key]) if row[key] not in ("", None) else None
        if "fold" in row:
            row["fold"] = int(row["fold"])
    return rows


@router.get("/models")
def model_comparison(
    results_dir: Annotated[Path, Depends(get_results_dir)],
    ctx: Annotated[AppContext, Depends(get_app_context)],
):
    """The latest saved LightGBM / Random Forest / Ridge / persistence comparison."""
    holdout_path = results_dir / "model_comparison.csv"
    holdout = _read_rows(holdout_path)
    if not holdout:
        raise HTTPException(
            status_code=404,
            detail="No comparison results yet. Run 'aiperfopt compare' to create them.",
        )
    settings = ctx.settings.forecasting
    return {
        "generated_at": holdout_path.stat().st_mtime,
        "source": str(holdout_path),
        "horizon_seconds": {
            "short": settings.short_horizon_target_seconds,
            "medium": settings.medium_horizon_target_seconds,
        },
        "holdout": holdout,
        "cv": _read_rows(results_dir / "model_comparison_cv.csv"),
        "regimes": _read_rows(results_dir / "paper" / "regimes.csv"),
    }
