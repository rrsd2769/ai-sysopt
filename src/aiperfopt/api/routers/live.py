from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from aiperfopt.api.connection_manager import ConnectionManager
from aiperfopt.api.dependencies import get_connection_manager, get_connection_manager_ws, get_live_loop, get_live_loop_ws
from aiperfopt.api.live_state import LiveInferenceLoop
from aiperfopt.api.schemas import LiveStateOut

router = APIRouter(tags=["live"])


@router.get("/live/state", response_model=LiveStateOut | None)
def get_live_state(loop: Annotated[LiveInferenceLoop, Depends(get_live_loop)]):
    """The most recent live-inference snapshot (a REST fallback for clients not using the WebSocket).

    Returns ``None`` if the live-inference loop hasn't completed a cycle
    yet (e.g. it was only just started, or was never started at all) --
    not an error; the API stays usable even with
    nothing to show yet.
    """
    return loop.latest_state()


@router.websocket("/ws/live")
async def live_websocket(
    websocket: WebSocket,
    manager: Annotated[ConnectionManager, Depends(get_connection_manager_ws)],
    loop: Annotated[LiveInferenceLoop, Depends(get_live_loop_ws)],
):
    """Streams every live-inference snapshot to this client as it's produced.

    On connect, immediately sends whatever the latest snapshot already
    is (if any) so a client doesn't have to wait a full
    ``live_inference_interval_seconds`` cycle to see anything -- then
    receives every subsequent broadcast pushed by the live-inference
    loop's ``on_update`` callback (wired in ``api.app``).
    """
    await manager.connect(websocket)
    try:
        current = loop.latest_state()
        if current is not None:
            await websocket.send_json(LiveStateOut.model_validate(current).model_dump(mode="json"))
        while True:
            # This endpoint is broadcast-only; receive() is used purely to
            # detect disconnects (a client message, if any, is ignored).
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket)
