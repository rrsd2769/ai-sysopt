"""
The API Layer entrypoint (Section 7.1 item 12). FR-12.4 requires this to
be the *only* access path to backend functionality — every route here
calls into an already-existing, already-tested subsystem function
(Phases 1-8); this module's own job is strictly wiring: build an
:class:`AppContext`, assemble the Monitoring pipeline and live-inference
loop around it, and mount the REST/WebSocket routers.

Nothing starts automatically on app creation
----------------------------------------------
Consistent with :func:`aiperfopt.bootstrap.bootstrap` and
:func:`aiperfopt.monitoring.pipeline.create_monitoring_pipeline` (neither of which
spawns a thread as a side effect of being called), :func:`create_app`
builds the Monitoring pipeline and live-inference loop but does not
start either — creating the FastAPI app (e.g. for a test client) must
never have the side effect of starting background threads. The lifespan
context only wires shutdown (stopping anything that *is* running when
the app closes); starting is always an explicit action, either via the
``/monitoring/start`` and ``/live/start`` endpoints or by passing
``auto_start=True`` to :func:`create_app` (used by the real server
entrypoint, :mod:`aiperfopt.api.server`).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from aiperfopt.api.connection_manager import ConnectionManager
from aiperfopt.api.live_state import LiveInferenceLoop, LiveState
from aiperfopt.api.routers import history, live, reporting, retraining, system
from aiperfopt.applog import get_logger
from aiperfopt.bootstrap import AppContext, bootstrap
from aiperfopt.features.config import FeatureEngineeringConfig
from aiperfopt.monitoring.pipeline import create_monitoring_pipeline
from aiperfopt.insight.catalog import load_recommendation_catalog

logger = get_logger(__name__)


def create_app(
    ctx: AppContext | None = None,
    user_config_path: Path | str | None = None,
    auto_start: bool = False,
    start_monitoring: bool = True,
) -> FastAPI:
    """Build a ready-to-serve FastAPI app.

    Parameters
    ----------
    ctx:
        An already-bootstrapped :class:`AppContext` (e.g. one a test
        fixture built against a temporary database). If omitted, this
        calls :func:`aiperfopt.bootstrap.bootstrap` itself.
    auto_start:
        If True, starts the live-inference loop immediately on FastAPI
        startup rather than waiting for an explicit ``/live/start`` call.
        The real server entrypoint wants this; a test client building an
        app just to exercise REST endpoints usually does not.
    start_monitoring:
        Whether ``auto_start`` also starts the Monitoring pipeline (the
        real collector polling this machine and writing to
        ``ctx.settings.storage.database_path``). Ignored unless
        ``auto_start`` is True. Set False when the configured database is
        a replay database (``StorageSettings.is_replay``) -- otherwise
        this bundled collector would write real, current-machine samples
        into the same file ``aiperfopt replay`` is writing historical
        ones into, exactly the concurrent-write hazard
        ``aiperfopt.replay.player.assert_monitoring_not_active`` exists to
        catch on the *primary* database. There is no equivalent guard for
        a server started this way, so the caller (``aiperfopt serve
        --no-monitor``) must opt out explicitly.
    """
    ctx = ctx or bootstrap(user_config_path)
    feature_config = FeatureEngineeringConfig()
    catalog = load_recommendation_catalog()
    monitoring_pipeline = create_monitoring_pipeline(ctx)
    connection_manager = ConnectionManager()

    live_loop = LiveInferenceLoop(
        storage_engine=ctx.storage_engine,
        feature_config=feature_config,
        forecasting_settings=ctx.settings.forecasting,
        contention_settings=ctx.settings.contention,
        diagnosis_settings=ctx.settings.diagnosis,
        retraining_settings=ctx.settings.retraining,
        catalog=catalog,
        interval_seconds=ctx.settings.api.live_inference_interval_seconds,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()

        def _broadcast(state: LiveState) -> None:
            if loop.is_closed():
                return
            payload_model = _live_state_to_schema(state)
            try:
                asyncio.run_coroutine_threadsafe(
                    connection_manager.broadcast(payload_model.model_dump(mode="json")), loop
                )
            except RuntimeError:
                # Loop closed between the check above and this call (shutdown race).
                pass

        live_loop.on_update = _broadcast

        if auto_start:
            if start_monitoring:
                monitoring_pipeline.start()
            live_loop.start()

        try:
            yield
        finally:
            live_loop.stop()
            monitoring_pipeline.stop()

    app = FastAPI(
        title="AI System Performance Optimizer API",
        description="Predictive, explainable macOS system health monitor -- API Layer (Phase 9, FR-12).",
        lifespan=lifespan,
    )
    app.state.ctx = ctx
    app.state.monitoring_pipeline = monitoring_pipeline
    app.state.live_loop = live_loop
    app.state.connection_manager = connection_manager
    app.state.catalog = catalog

    app.include_router(system.router)
    app.include_router(history.router)
    app.include_router(retraining.router)
    app.include_router(live.router)
    app.include_router(reporting.router)

    return app


def _live_state_to_schema(state: LiveState):
    from aiperfopt.api.schemas import LiveStateOut

    return LiveStateOut.model_validate(state)
