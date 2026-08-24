"""
Tracks connected WebSocket clients and broadcasts each live-inference
snapshot to all of them. Deliberately minimal: no per-client filtering
or backpressure handling beyond "drop a connection that errors on
send" — Section 5's single-machine, single-user V1 scope means this
never needs to handle more than a handful of concurrent local
connections (NFR-6.1).
"""

from __future__ import annotations

import asyncio

from fastapi import WebSocket

from aiperfopt.applog import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info("WebSocket client connected", extra={"connection_count": len(self._connections)})

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)
        logger.info("WebSocket client disconnected", extra={"connection_count": len(self._connections)})

    async def broadcast(self, message: dict) -> None:
        async with self._lock:
            connections = list(self._connections)
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception:  # noqa: BLE001 - a dead connection must not break the broadcast for others
                await self.disconnect(websocket)
