from __future__ import annotations

import time
from pathlib import Path

from aiperfopt.frontend.config import FrontendSettings
from aiperfopt.frontend.live_client import LiveWebSocketClient
from aiperfopt.storage import insert_system_metric_samples_batch
from tests.api_conftest import build_test_ctx, rising_cpu_samples_now
from tests.frontend.server_fixture import running_backend


def _settings_for(base_url: str) -> FrontendSettings:
    host, port = base_url.replace("http://", "").split(":")
    return FrontendSettings(api_host=host, api_port=int(port))


def test_receives_broadcast_snapshot(qtbot, tmp_path: Path):
    ctx = build_test_ctx(tmp_path, live_inference_interval_seconds=0.5)
    samples = rising_cpu_samples_now(n=300, interval=2.0)
    insert_system_metric_samples_batch(ctx.storage_engine, samples)

    with running_backend(tmp_path, ctx=ctx, auto_start=True) as (base_url, _ctx):
        client = LiveWebSocketClient(settings=_settings_for(base_url))
        received: list[dict] = []
        client.state_received.connect(received.append)

        with qtbot.waitSignal(client.connection_changed, timeout=5000):
            client.start()

        # Wait for at least one broadcast beyond the immediate on-connect send.
        qtbot.waitUntil(lambda: len(received) >= 1, timeout=5000)
        assert "timestamp" in received[0]

        client.stop()


def test_connection_changed_signal_reflects_state(qtbot, tmp_path: Path):
    with running_backend(tmp_path) as (base_url, _ctx):
        client = LiveWebSocketClient(settings=_settings_for(base_url))
        states: list[bool] = []
        client.connection_changed.connect(states.append)

        with qtbot.waitSignal(client.connection_changed, timeout=5000):
            client.start()
        assert client.is_connected is True

        client.stop()
        assert states[0] is True


def test_reconnects_after_backend_restarts(qtbot, tmp_path: Path):
    """The client should recover once a backend it lost becomes reachable again."""
    ctx = build_test_ctx(tmp_path)
    client = LiveWebSocketClient(settings=FrontendSettings(api_host="127.0.0.1", api_port=1))
    client.start()

    # No server at all yet -- give the retry timer a couple of cycles to
    # prove it doesn't crash/give up.
    qtbot.wait(2500)
    assert client.is_connected is False

    client.stop()
