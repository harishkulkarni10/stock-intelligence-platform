"""Unit tests for structured JSON logging."""

from __future__ import annotations

import json
import logging

from logger.context import bind_context, clear_context
from logger.logger import JsonFormatter, configure_logging, get_logger, log_event


def test_json_formatter_includes_context_and_metrics():
    configure_logging()
    clear_context()
    bind_context(trace_id="req-test123", ticker="MSFT", request_path="/analyze")
    try:
        logger = get_logger("sip.test")
        record = logger.makeRecord(
            name="sip.test",
            level=logging.INFO,
            fn="test",
            lno=1,
            msg="agent1_validation",
            args=(),
            exc_info=None,
        )
        # Filters normally attach context; simulate what ContextFilter does.
        from logger.logger import ContextFilter

        ContextFilter().filter(record)
        record.step = "agent1_validation"
        record.status = "success"
        record.duration_ms = 4820
        record.metrics = {"attempt": 1, "chars": 1273}
        record.data = {"trend": "SIDEWAYS", "errors": []}
        line = JsonFormatter().format(record)
        payload = json.loads(line)
        assert payload["trace_id"] == "req-test123"
        assert payload["ticker"] == "MSFT"
        assert payload["step"] == "agent1_validation"
        assert payload["duration_ms"] == 4820
        assert payload["metrics"]["chars"] == 1273
        assert payload["data"]["trend"] == "SIDEWAYS"
        assert "timestamp" in payload
    finally:
        clear_context()


def test_log_event_emits_without_error(caplog):
    configure_logging()
    logger = get_logger("sip.test")
    with caplog.at_level(logging.INFO, logger="sip.test"):
        log_event(
            logger,
            "probe",
            step="unit",
            status="ok",
            duration_ms=12.5,
            metrics={"n": 1},
        )
    assert True
