"""
Read-only historical query endpoints. Every one of these is a thin
wrapper over an already-existing, already-tested
:mod:`aiperfopt.storage.reader` function — historical
queries are exposed via REST; this layer does not invent new query logic.
Query parameters (``start``, ``end``, ``limit``) map directly onto each
reader function's own parameters.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Engine

from aiperfopt.api.dependencies import get_storage_engine
from aiperfopt.api.schemas import (
    ContentionPredictionOut,
    DiagnosisOut,
    EvaluationOut,
    FeatureVectorOut,
    ForecastPredictionOut,
    ModelDeploymentOut,
    RecommendationOut,
    SystemMetricSampleOut,
)
from aiperfopt.storage.reader import (
    get_contention_predictions_range,
    get_diagnoses_range,
    get_evaluations_by_version,
    get_evaluations_range,
    get_features_range,
    get_forecast_predictions_range,
    get_latest_model_deployment,
    get_model_deployments_range,
    get_recommendations_range,
    get_system_metrics_range,
)

router = APIRouter(tags=["history"])

_Engine = Annotated[Engine, Depends(get_storage_engine)]
_Start = Annotated[float | None, Query(description="Inclusive start timestamp (epoch seconds)")]
_End = Annotated[float | None, Query(description="Inclusive end timestamp (epoch seconds)")]
_Limit = Annotated[int | None, Query(gt=0, description="Maximum rows to return")]


@router.get("/metrics/system", response_model=list[SystemMetricSampleOut])
def system_metrics(engine: _Engine, start: _Start = None, end: _End = None, limit: _Limit = None):
    return get_system_metrics_range(engine, start=start, end=end, limit=limit)


@router.get("/features", response_model=list[FeatureVectorOut])
def features(engine: _Engine, start: _Start = None, end: _End = None, limit: _Limit = None):
    return get_features_range(engine, start=start, end=end, limit=limit)


@router.get("/forecasts", response_model=list[ForecastPredictionOut])
def forecasts(engine: _Engine, start: _Start = None, end: _End = None, limit: _Limit = None):
    return get_forecast_predictions_range(engine, start=start, end=end, limit=limit)


@router.get("/contention", response_model=list[ContentionPredictionOut])
def contention_predictions(engine: _Engine, start: _Start = None, end: _End = None, limit: _Limit = None):
    return get_contention_predictions_range(engine, start=start, end=end, limit=limit)


@router.get("/diagnoses", response_model=list[DiagnosisOut])
def diagnoses(engine: _Engine, start: _Start = None, end: _End = None, limit: _Limit = None):
    return get_diagnoses_range(engine, start=start, end=end, limit=limit)


@router.get("/recommendations", response_model=list[RecommendationOut])
def recommendations(engine: _Engine, start: _Start = None, end: _End = None, limit: _Limit = None):
    return get_recommendations_range(engine, start=start, end=end, limit=limit)


@router.get("/evaluations", response_model=list[EvaluationOut])
def evaluations(engine: _Engine, start: _Start = None, end: _End = None, limit: _Limit = None):
    return get_evaluations_range(engine, start=start, end=end, limit=limit)


@router.get("/evaluations/by-version/{model_version}", response_model=list[EvaluationOut])
def evaluations_by_version(model_version: str, engine: _Engine):
    return get_evaluations_by_version(engine, model_version)


@router.get("/deployments", response_model=list[ModelDeploymentOut])
def deployments(engine: _Engine, start: _Start = None, end: _End = None, limit: _Limit = None):
    return get_model_deployments_range(engine, start=start, end=end, limit=limit)


@router.get("/deployments/latest", response_model=ModelDeploymentOut | None)
def latest_deployment(engine: _Engine):
    return get_latest_model_deployment(engine)
