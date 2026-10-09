"""
Reporting endpoints (Section 7.1 item 14, FR-14.1/FR-14.2).

Thin wrapper over :mod:`aiperfopt.reporting` -- matching the pattern
every other subsystem's router already follows (e.g.
``routers/retraining.py`` over ``aiperfopt.lifecycle``): the actual
join/aggregation logic lives in the subsystem package, not here.

- ``GET /reporting/predicted-vs-actual`` -- FR-13.3's per-metric
  predicted-vs-actual comparison view (built in Phase 10; the join logic
  itself moved into ``aiperfopt.reporting.predicted_vs_actual`` in Phase
  11 once ``/reporting/summary`` below became a second consumer of it).
- ``GET /reporting/summary`` -- FR-14.1's exportable daily/weekly
  summary, combining system-metrics context with prediction-accuracy
  (FR-14.2), contention, diagnosis, and recommendation summaries
  (``aiperfopt.reporting.summary.build_report``). ``?download=true``
  returns the same JSON body as a downloadable attachment rather than an
  inline response -- satisfying FR-14.1's "exportable" without inventing
  a second format (Section 18 defers format specifics; JSON-over-REST is
  the project's existing default everywhere else in the API).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import Engine

from aiperfopt.api.dependencies import get_storage_engine
from aiperfopt.api.schemas import PredictedVsActualPointOut, ReportSummaryOut
from aiperfopt.reporting.predicted_vs_actual import VALID_METRIC_NAMES, join_predicted_vs_actual
from aiperfopt.reporting.summary import build_report

router = APIRouter(prefix="/reporting", tags=["reporting"])

_Engine = Annotated[Engine, Depends(get_storage_engine)]


@router.get("/predicted-vs-actual", response_model=list[PredictedVsActualPointOut])
def predicted_vs_actual(
    engine: _Engine,
    metric_name: Annotated[str, Query(description="A raw_system_metrics column, e.g. 'cpu_percent'")],
    start: Annotated[float | None, Query(description="Inclusive start timestamp (epoch seconds)")] = None,
    end: Annotated[float | None, Query(description="Inclusive end timestamp (epoch seconds)")] = None,
    horizon_label: Annotated[str | None, Query(description="Restrict to one horizon, e.g. 'short'")] = None,
    limit: Annotated[int | None, Query(gt=0, description="Maximum predictions to return")] = None,
):
    """Join forecast predictions against the actual metric value nearest each ``target_timestamp``.

    Powers FR-13.3's predicted-vs-actual comparison view: for every
    forecast this model/metric produced in range, this returns the
    predicted value alongside whatever was actually observed closest to
    the moment that prediction was targeting -- ``actual_value`` is
    ``None`` when no observed sample falls close enough to the target
    (e.g. the target is in the near future and hasn't happened yet, or
    monitoring had a gap), which a client should render as "not yet
    available", not an error.
    """
    try:
        points = join_predicted_vs_actual(
            engine, metric_name, start=start, end=end, horizon_label=horizon_label, limit=limit
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return [
        PredictedVsActualPointOut(
            metric_name=p.metric_name, horizon_label=p.horizon_label, generated_at=p.generated_at,
            target_timestamp=p.target_timestamp, predicted_value=p.predicted_value,
            actual_value=p.actual_value, error=p.error, confidence=p.confidence,
            model_version=p.model_version,
        )
        for p in points
    ]


@router.get("/summary", response_model=ReportSummaryOut)
def summary(
    engine: _Engine,
    period: Annotated[str | None, Query(description="'daily' or 'weekly'; ignored if start and end are both given")] = None,
    start: Annotated[float | None, Query(description="Inclusive start timestamp (epoch seconds)")] = None,
    end: Annotated[float | None, Query(description="Inclusive end timestamp (epoch seconds); defaults to now")] = None,
    download: Annotated[bool, Query(description="Return as a downloadable attachment (FR-14.1)")] = False,
):
    """An exportable report combining system-metrics context with prediction
    accuracy (FR-14.2), contention, diagnosis, and recommendation summaries
    over a window (FR-14.1) -- either a named ``period`` ("daily"/"weekly")
    ending now/at ``end``, or an explicit ``start``/``end`` pair.
    """
    try:
        report = build_report(engine, period=period, start=start, end=end)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    body = ReportSummaryOut.model_validate(report, from_attributes=True)
    if not download:
        return body

    filename = f"aiperfopt_report_{int(report.start)}_{int(report.end)}.json"
    return JSONResponse(
        content=body.model_dump(mode="json"),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/metrics", response_model=list[str])
def valid_metric_names() -> list[str]:
    """The metric names ``/predicted-vs-actual`` accepts -- lets a client
    (e.g. Phase 10's metric picker) build its options from the backend's
    actual forecasting target set instead of a hardcoded duplicate list.
    """
    return sorted(VALID_METRIC_NAMES)
