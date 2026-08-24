from __future__ import annotations

import json
import logging
from pathlib import Path

from aiperfopt.applog import configure_logging, get_logger
from aiperfopt.applog.formatting import JsonFormatter
from aiperfopt.config.settings import LoggingSettings


def test_json_formatter_produces_valid_json():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="aiperfopt.monitoring", level=logging.INFO, pathname=__file__,
        lineno=1, msg="collected sample", args=(), exc_info=None,
    )
    line = formatter.format(record)
    parsed = json.loads(line)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "aiperfopt.monitoring"
    assert parsed["message"] == "collected sample"
    assert "timestamp" in parsed


def test_json_formatter_includes_extra_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="aiperfopt.storage", level=logging.WARNING, pathname=__file__,
        lineno=1, msg="write slow", args=(), exc_info=None,
    )
    record.database_path = "/tmp/aiperfopt.db"
    record.latency_ms = 812

    parsed = json.loads(formatter.format(record))

    assert parsed["database_path"] == "/tmp/aiperfopt.db"
    assert parsed["latency_ms"] == 812


def test_json_formatter_includes_exception_info():
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="aiperfopt.forecasting", level=logging.ERROR, pathname=__file__,
            lineno=1, msg="forecast failed", args=(), exc_info=sys.exc_info(),
        )
    parsed = json.loads(formatter.format(record))

    assert parsed["exception"]["type"] == "ValueError"
    assert parsed["exception"]["message"] == "boom"
    assert "traceback" in parsed["exception"]


def test_configure_logging_writes_json_lines_to_file(tmp_path: Path):
    settings = LoggingSettings(level="INFO", log_dir=tmp_path, json_format=True)
    configure_logging(settings, force=True)

    logger = get_logger("aiperfopt.test_stage")
    logger.info("hello from a pipeline stage", extra={"stage": "test"})

    for handler in logging.getLogger("aiperfopt").handlers:
        handler.flush()

    log_file = tmp_path / "aiperfopt.log"
    assert log_file.exists()
    lines = [line for line in log_file.read_text().splitlines() if line.strip()]
    assert len(lines) >= 1
    parsed = json.loads(lines[-1])
    assert parsed["message"] == "hello from a pipeline stage"
    assert parsed["stage"] == "test"


def test_configure_logging_is_idempotent_without_force(tmp_path: Path):
    settings = LoggingSettings(level="INFO", log_dir=tmp_path, json_format=True)
    configure_logging(settings, force=True)
    handler_count_after_first = len(logging.getLogger("aiperfopt").handlers)

    configure_logging(settings)  # no force -> should be a no-op
    handler_count_after_second = len(logging.getLogger("aiperfopt").handlers)

    assert handler_count_after_first == handler_count_after_second


def test_configure_logging_respects_level(tmp_path: Path):
    settings = LoggingSettings(level="ERROR", log_dir=tmp_path, json_format=True)
    configure_logging(settings, force=True)

    logger = get_logger("aiperfopt.test_stage_2")
    logger.info("should be suppressed")
    logger.error("should appear")

    for handler in logging.getLogger("aiperfopt").handlers:
        handler.flush()

    log_file = tmp_path / "aiperfopt.log"
    content = log_file.read_text()
    assert "should be suppressed" not in content
    assert "should appear" in content
