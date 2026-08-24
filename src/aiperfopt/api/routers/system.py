from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from aiperfopt.api.dependencies import get_app_context, get_live_loop, get_monitoring_pipeline
from aiperfopt.bootstrap import AppContext
from aiperfopt.api.live_state import LiveInferenceLoop
from aiperfopt.api.schemas import ReadinessOut
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.pipeline import MonitoringPipeline
from aiperfopt.readiness import assess_readiness

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/config")
def get_config(ctx: Annotated[AppContext, Depends(get_app_context)]) -> dict:
    """Read-only view of the currently active configuration (NFR-12.1).

    Returned as a plain dict rather than re-exposing pydantic-settings
    objects directly — this is a read-only diagnostic view, not a config
    write path (there is no corresponding PUT/PATCH; FR-15.1's layered
    file/env config remains the only way to change settings, consistent
    with V1's local-only, no-remote-admin scope, Section 13).
    """
    return ctx.settings.model_dump(mode="json")


@router.get("/readiness", response_model=ReadinessOut)
def readiness(ctx: Annotated[AppContext, Depends(get_app_context)]) -> ReadinessOut:
    """Why the live pipeline can or cannot produce a prediction right now.

    The frontend renders this instead of a hardcoded "Live" banner, so a
    system that is merely still collecting data no longer looks identical
    to one that is working.
    """
    result = assess_readiness(ctx.storage_engine, FeatureEngineeringConfig(), ctx.settings.retraining)
    return ReadinessOut(
        state=result.state.value,
        detail=result.detail,
        history_seconds=result.history_seconds,
        history_seconds_required=result.history_seconds_required,
        active_model_version=result.active_model_version,
        can_predict=result.can_predict,
    )


@router.get("/monitoring/status")
def monitoring_status(pipeline: Annotated[MonitoringPipeline, Depends(get_monitoring_pipeline)]) -> dict:
    return {"running": pipeline.engine.is_running}


@router.post("/monitoring/start")
def start_monitoring(pipeline: Annotated[MonitoringPipeline, Depends(get_monitoring_pipeline)]) -> dict:
    pipeline.start()
    return {"running": True}


@router.post("/monitoring/stop")
def stop_monitoring(pipeline: Annotated[MonitoringPipeline, Depends(get_monitoring_pipeline)]) -> dict:
    pipeline.stop()
    return {"running": False}


@router.get("/live/status")
def live_status(loop: Annotated[LiveInferenceLoop, Depends(get_live_loop)]) -> dict:
    return {"running": loop.is_running}


@router.post("/live/start")
def start_live_inference(loop: Annotated[LiveInferenceLoop, Depends(get_live_loop)]) -> dict:
    loop.start()
    return {"running": True}


@router.post("/live/stop")
def stop_live_inference(loop: Annotated[LiveInferenceLoop, Depends(get_live_loop)]) -> dict:
    loop.stop()
    return {"running": False}
