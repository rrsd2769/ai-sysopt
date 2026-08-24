from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from aiperfopt.api.dependencies import get_app_context, get_live_loop, get_monitoring_pipeline
from aiperfopt.bootstrap import AppContext
from aiperfopt.api.live_state import LiveInferenceLoop
from aiperfopt.pipeline import MonitoringPipeline

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
