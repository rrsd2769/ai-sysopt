"""
Runs a single blocking callable (typically an ``ApiClient`` method) on
``QThreadPool`` and delivers its result back via a Qt signal.

Exists specifically because of the fact that
``/retraining/run`` blocks
synchronously for the duration of training (tens of seconds to a few
minutes). A "Retrain now" button in Developer Mode must
not freeze the whole UI for that entire window -- this is the
background-job pattern that note calls for, applied client-side (the
backend endpoint itself is unchanged; the frontend just never calls it
from the UI thread directly).

Used for every REST call this frontend makes, not just retraining, so
one pattern covers both "a slow request shouldn't freeze the window" and
"a merely-not-instant local request shouldn't cause a visible stutter."
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _WorkerSignals(QObject):
    succeeded = Signal(object)
    failed = Signal(Exception)


class _CallableRunnable(QRunnable):
    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self._fn = fn
        self.signals = _WorkerSignals()
        # ``QRunnable`` auto-deletes itself on the *worker* thread right
        # after ``run()`` returns (the default). That would drop the last
        # Python reference to ``self.signals`` -- a QObject affine to the
        # UI thread -- from the worker thread, destroying it there instead
        # of on its own thread. That race is a real, intermittent SIGSEGV
        # source (same class of bug as the unparented QWebSocket fix
        # elsewhere in this module's sibling ``live_client.py``), observed
        # here as flaky crashes in ``tests/frontend/test_retraining_panel.py``.
        # We manage the runnable's lifetime ourselves instead, via
        # ``_schedule_signals_cleanup`` below.
        self.setAutoDelete(False)

    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller via a signal, not swallowed
            self.signals.failed.emit(exc)
        else:
            self.signals.succeeded.emit(result)


def run_in_background(
    fn: Callable[[], Any],
    on_success: Callable[[Any], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
    pool: QThreadPool | None = None,
) -> None:
    """Fire-and-forget: run ``fn()`` off the UI thread; deliver the outcome back on it.

    Qt's queued-connection signal delivery (the default across threads)
    is what gets ``on_success``/``on_error`` safely back onto the UI
    thread -- no manual locking needed for this simple "one call, one
    result" shape. The runnable and its signals object are only ever torn
    down via ``deleteLater()``, triggered from a slot that itself only
    runs after delivery on the UI thread -- so destruction always happens
    on the thread the signals object lives on, never the worker thread.
    """
    runnable = _CallableRunnable(fn)
    if on_success is not None:
        runnable.signals.succeeded.connect(on_success)
    if on_error is not None:
        runnable.signals.failed.connect(on_error)

    def _cleanup(*_args: Any) -> None:
        runnable.signals.deleteLater()

    runnable.signals.succeeded.connect(_cleanup)
    runnable.signals.failed.connect(_cleanup)
    (pool or QThreadPool.globalInstance()).start(runnable)
