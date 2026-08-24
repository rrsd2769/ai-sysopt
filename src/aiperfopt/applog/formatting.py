"""
JSON log formatter.

Implements FR-16.1 (structured JSON logs per pipeline stage, separable
from metric/training data storage) using the Python standard library's
``logging`` module plus a custom ``Formatter`` — per Section 19's
technology choice of stdlib logging + JSON formatter over structlog
(marginal benefit over stdlib for this project's scope).
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, ClassVar

# Attributes present on every LogRecord that we do NOT want to re-emit as
# "extra" fields, since they're either already surfaced explicitly
# (message, level, timestamp, logger) or are CPython implementation detail
# irrelevant to a structured log consumer.
_STANDARD_RECORD_ATTRS = frozenset(
    logging.LogRecord(
        name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
    ).__dict__.keys()
) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Renders each log record as a single JSON object per line.

    Every record includes: timestamp (ISO-8601 UTC), level, logger name
    (used as the "pipeline stage" per FR-16.1 — e.g. ``aiperfopt.monitoring``),
    the message, and, when present, exception info and any structured
    ``extra=`` fields passed at the call site (e.g. ``context`` from an
    :class:`~aiperfopt.common.exceptions.AppError`).
    """

    # Reserved key names that always appear first for readability when a
    # human tails the log file, even though ordering has no semantic
    # meaning for JSON.
    _CORE_KEYS: ClassVar[tuple[str, ...]] = ("timestamp", "level", "logger", "message")

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Structured extras (anything passed via `logging.info(..., extra={...})`)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in payload:
                payload[key] = _json_safe(value)

        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": "".join(traceback.format_exception(*record.exc_info)),
            }

        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=_json_safe)


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of arbitrary objects into JSON-serializable form.

    Never raises — worst case, an un-serializable value is stringified,
    which keeps a single bad log call from breaking the logging pipeline
    itself (consistent with the project-wide error-handling philosophy).
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


class PlainFormatter(logging.Formatter):
    """Human-readable formatter used when ``json_format`` is disabled.

    Intended for local interactive development, where a JSON-per-line log
    is harder to read than the classic ``TIMESTAMP LEVEL logger: message``
    layout. Not used in production/demo runs per the default configuration.
    """

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
