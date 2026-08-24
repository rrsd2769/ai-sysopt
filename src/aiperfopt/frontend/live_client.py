"""
Streams ``/ws/live`` broadcasts (FR-12.1) into the Qt event loop.

Uses ``PySide6.QtWebSockets.QWebSocket`` rather than a second background
thread pushing through queues/signals -- ``QWebSocket`` already integrates
with Qt's own event loop, so there's no synchronization primitive to
build here the way the Monitoring Engine needed one (ADL-18's
thread+queue pattern solves a different problem: getting *out* of a
non-Qt polling thread; this client never leaves the Qt event loop in the
first place).

Reconnection is deliberately simple: on any disconnect or error, wait a
fixed backoff and try again, indefinitely, via a ``QTimer``. FR-13's
"consume only through the API" scope doesn't include reasoning about
backend restarts specially -- the *frontend's* job under NFR-11.2 ("if
the backend/Monitoring is unavailable, the UI remains usable") is simply
to keep trying and let widgets keep showing their last-known state in
the meantime, not to implement anything fancier (exponential backoff,
circuit breaking) that V1's local-only, single-user scope doesn't need.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtWebSockets import QWebSocket

from aiperfopt.applog import get_logger
from aiperfopt.frontend.config import FrontendSettings

logger = get_logger(__name__)

_RECONNECT_INTERVAL_MS = 2000


class LiveWebSocketClient(QObject):
    """Connects to ``/ws/live`` and emits every decoded snapshot.

    Signals
    -------
    state_received(dict):
        One decoded ``LiveStateOut`` JSON payload, emitted for every
        message the server broadcasts (including the immediate
        "here's the latest snapshot" message sent right on connect).
    connection_changed(bool):
        Emitted whenever the socket transitions connected/disconnected,
        so a status indicator can reflect it without polling.
    """

    state_received = Signal(dict)
    connection_changed = Signal(bool)

    def __init__(self, settings: FrontendSettings | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.settings = settings or FrontendSettings()
        self._socket = QWebSocket()
        self._socket.connected.connect(self._on_connected)
        self._socket.disconnected.connect(self._on_disconnected)
        self._socket.errorOccurred.connect(self._on_error)
        self._socket.textMessageReceived.connect(self._on_text_message)

        self._reconnect_timer = QTimer(self)
        self._reconnect_timer.setInterval(_RECONNECT_INTERVAL_MS)
        self._reconnect_timer.timeout.connect(self._attempt_connect)

        self._wanted = False  # True once start() is called, until stop()
        self.is_connected = False

    def start(self) -> None:
        """Begin (re)connecting. Safe to call once; subsequent drops auto-retry."""
        self._wanted = True
        self._attempt_connect()
        self._reconnect_timer.start()

    def stop(self) -> None:
        """Stop trying to connect and close any open connection."""
        self._wanted = False
        self._reconnect_timer.stop()
        self._socket.close()

    def _attempt_connect(self) -> None:
        if not self._wanted or self.is_connected:
            return
        self._socket.open(QUrl(self.settings.ws_url))

    def _on_connected(self) -> None:
        self.is_connected = True
        self.connection_changed.emit(True)

    def _on_disconnected(self) -> None:
        was_connected = self.is_connected
        self.is_connected = False
        if was_connected:
            self.connection_changed.emit(False)

    def _on_error(self, _error) -> None:
        logger.warning("live websocket error", extra={"error": self._socket.errorString()})

    def _on_text_message(self, message: str) -> None:
        import json

        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("received non-JSON live websocket message", extra={"message": message[:200]})
            return
        self.state_received.emit(payload)
