"""
Logging initialization for the AI System Performance Optimizer.

Call :func:`configure_logging` exactly once, near process startup (before
any subsystem is constructed), then have every module obtain its logger
via :func:`get_logger` using its module's dotted name. Using the module
name as the logger name means the logger name itself doubles as the
"pipeline stage" tag required by FR-16.1 — e.g. records from
``aiperfopt.monitoring.engine`` are trivially filterable by stage without
any extra bookkeeping.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from aiperfopt.applog.formatting import JsonFormatter, PlainFormatter
from aiperfopt.common.exceptions import LoggingSetupError
from aiperfopt.config.settings import LoggingSettings

# Root logger namespace for the whole application. Subsystems log under
# "aiperfopt.<subsystem>" (via __name__), which is a child of this logger
# by Python logging's dot-hierarchy, so configuring handlers/level here
# once is sufficient for the entire pipeline.
_ROOT_LOGGER_NAME = "aiperfopt"

_configured = False


def configure_logging(settings: LoggingSettings, *, force: bool = False) -> None:
    """Attach console + rotating-file handlers to the ``aiperfopt`` logger tree.

    Idempotent by default: calling this more than once (e.g. because both
    a CLI entrypoint and a test fixture call it) is a no-op after the
    first call, unless ``force=True`` is passed, which tears down and
    reattaches handlers. This avoids duplicate log lines from stacking
    handlers on repeated calls.

    Parameters
    ----------
    settings:
        The logging section of :class:`~aiperfopt.config.settings.AppSettings`.
    force:
        Reconfigure even if already configured. Primarily useful in tests
        that need a fresh handler set per test case.

    Raises
    ------
    LoggingSetupError
        If the log directory cannot be created or a handler cannot be
        attached.
    """
    global _configured
    if _configured and not force:
        return

    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(settings.level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter: logging.Formatter = JsonFormatter() if settings.json_format else PlainFormatter()

    try:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        log_dir = Path(settings.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "aiperfopt.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        raise LoggingSetupError(
            "Failed to initialize logging handlers",
            recoverable=False,
            context={"log_dir": str(settings.log_dir)},
        ) from exc

    _configured = True
    logger.debug(
        "Logging configured", extra={"level": settings.level, "json_format": settings.json_format}
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger for ``name`` (conventionally ``__name__``).

    If :func:`configure_logging` has not yet been called, the returned
    logger falls back to Python logging's default behavior (handler-less,
    propagates to the root logger with the standard "no handlers" warning
    on first use). Subsystems should not call this before startup
    configuration in normal operation, but this fallback keeps ad hoc
    scripts and early unit tests from crashing.
    """
    return logging.getLogger(name)


def reset_logging_for_tests() -> None:
    """Tear down handlers and reset configuration state.

    Intended for use in test fixtures only, so each test can call
    :func:`configure_logging` against a fresh temporary log directory
    without handlers leaking across test cases.
    """
    global _configured
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    _configured = False
