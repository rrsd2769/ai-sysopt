"""
FastAPI dependency providers. Every route handler reaches the rest of
the application (storage engine, settings, the live-inference loop)
through these — never by importing a global — so the same route module
works against a fresh test app with its own isolated context
as it does against the real server process.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, WebSocket
from sqlalchemy import Engine

from aiperfopt.api.connection_manager import ConnectionManager
from aiperfopt.api.live_state import LiveInferenceLoop
from aiperfopt.bootstrap import AppContext
from aiperfopt.monitoring.pipeline import MonitoringPipeline


def get_app_context(request: Request) -> AppContext:
    return request.app.state.ctx


def get_storage_engine(ctx: Annotated[AppContext, Depends(get_app_context)]) -> Engine:
    return ctx.storage_engine


def get_monitoring_pipeline(request: Request) -> MonitoringPipeline:
    return request.app.state.monitoring_pipeline


def get_live_loop(request: Request) -> LiveInferenceLoop:
    return request.app.state.live_loop


def get_connection_manager(request: Request) -> ConnectionManager:
    return request.app.state.connection_manager


def get_live_loop_ws(websocket: WebSocket) -> LiveInferenceLoop:
    """WebSocket-route counterpart to :func:`get_live_loop`.

    FastAPI resolves dependencies differently for WebSocket routes than
    HTTP routes -- a dependency typed with ``Request`` cannot be reused
    there, since no ``Request`` exists in a WebSocket connection's scope
    (only a ``WebSocket``). Both reach the same ``app.state``.
    """
    return websocket.app.state.live_loop


def get_connection_manager_ws(websocket: WebSocket) -> ConnectionManager:
    return websocket.app.state.connection_manager
