"""
Runs a real backend (the FastAPI app under real uvicorn, on a real
socket) on a background thread for frontend tests to talk to.

Not named ``conftest.py``, matching ``tests/api_conftest.py``'s existing
precedent -- imported explicitly by the frontend test files that need a
real server rather than auto-discovered for the whole suite.

This mirrors the project's established "real threads, no mocking"
testing discipline (Monitoring/Ingestion
tests use real background threads, not mocked timers) applied to the UI: ``ApiClient`` is a real synchronous HTTP client and ``QWebSocket`` is
a real WebSocket client, so what they need to talk to is a real HTTP/WS
server, not an ASGI in-process shortcut (httpx's ``ASGITransport`` is
async-only and can't back a sync ``httpx.Client`` the way
``fastapi.testclient.TestClient`` fakes around internally -- a real
socket sidesteps that entirely and is arguably more representative of
what the shipped frontend actually does).
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from aiperfopt.api.app import create_app
from aiperfopt.bootstrap import AppContext
from tests.api_conftest import build_test_ctx


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def running_backend(tmp_path: Path, ctx: AppContext | None = None, auto_start: bool = False):
    """Yields ``(base_url, ctx)`` for a live backend serving on a real port.

    ``ctx`` may be pre-populated with history before the server starts
    (mirroring ``api_conftest.build_test_client``'s ``with_history``
    pattern) by building it via :func:`aiperfopt.frontend...api_conftest.build_test_ctx`
    and passing it in.
    """
    ctx = ctx or build_test_ctx(tmp_path)
    port = _free_port()
    app = create_app(ctx, auto_start=auto_start)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            httpx.get(f"{base_url}/health", timeout=0.5)
            break
        except httpx.TransportError:
            time.sleep(0.05)
    else:
        raise RuntimeError("Test backend did not become ready in time")

    try:
        yield base_url, ctx
    finally:
        app.state.live_loop.stop()
        app.state.monitoring_pipeline.stop()
        server.should_exit = True
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise RuntimeError("Test backend thread did not shut down in time")
