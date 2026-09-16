"""Production-ready SIP logger: JSON by default, context fields, event helper."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from logger.context import get_request_path, get_ticker, get_trace_id

_CONFIGURED = False


class ContextFilter(logging.Filter):
    """Inject trace_id / ticker / path from contextvars into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id() or "-"
        record.ticker = get_ticker() or "-"
        record.request_path = get_request_path() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line for CloudWatch / ELK / Datadog."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "-"),
            "ticker": getattr(record, "ticker", "-"),
        }
        path = getattr(record, "request_path", None)
        if path and path != "-":
            payload["request_path"] = path

        for key in ("step", "status", "event", "metrics", "data", "duration_ms"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    """Human-readable, no ANSI colors — safe for log collectors."""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, self.datefmt)} | {record.levelname} | "
            f"{record.name} | trace_id={getattr(record, 'trace_id', '-')} | "
            f"ticker={getattr(record, 'ticker', '-')} | {record.getMessage()}"
        )
        extras: list[str] = []
        for key in ("step", "status", "duration_ms"):
            value = getattr(record, key, None)
            if value is not None:
                extras.append(f"{key}={value}")
        metrics = getattr(record, "metrics", None)
        if isinstance(metrics, dict) and metrics:
            extras.append("metrics=" + json.dumps(metrics, default=str))
        data = getattr(record, "data", None)
        if isinstance(data, dict) and data:
            extras.append("data=" + json.dumps(data, default=str))
        if extras:
            base = f"{base} | " + " | ".join(extras)
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def _formatter() -> logging.Formatter:
    mode = (os.getenv("LOG_FORMAT") or "json").strip().lower()
    if mode in {"text", "plain", "human"}:
        return PlainFormatter(
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    return JsonFormatter()


def configure_logging() -> None:
    """Idempotent setup for sip + uvicorn loggers (no ANSI colors)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    # Discourage colored uvicorn / click output in collectors.
    os.environ.setdefault("NO_COLOR", "1")
    os.environ.setdefault("TERM", "dumb")

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    formatter = _formatter()
    context_filter = ContextFilter()

    def _attach(logger: logging.Logger, *, to_file: bool) -> None:
        logger.handlers.clear()
        logger.setLevel(level)
        logger.propagate = False
        logger.addFilter(context_filter)

        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        stream.addFilter(context_filter)
        logger.addHandler(stream)

        if to_file:
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log",
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(context_filter)
            logger.addHandler(file_handler)

    _attach(logging.getLogger("sip"), to_file=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        _attach(logging.getLogger(name), to_file=False)


def get_logger(name: str = "sip") -> logging.Logger:
    configure_logging()
    if name == "sip" or name.startswith("sip."):
        return logging.getLogger(name if name != "sip" else "sip")
    return logging.getLogger(f"sip.{name}" if not name.startswith("sip") else name)


def log_event(
    logger: logging.Logger,
    message: str,
    *,
    level: int = logging.INFO,
    step: str | None = None,
    status: str | None = None,
    metrics: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    duration_ms: float | int | None = None,
    exc_info: bool = False,
) -> None:
    """Emit a structured event; fields become JSON keys under LOG_FORMAT=json."""
    extra: dict[str, Any] = {}
    if step is not None:
        extra["step"] = step
    if status is not None:
        extra["status"] = status
    if metrics is not None:
        extra["metrics"] = metrics
    if data is not None:
        extra["data"] = data
    if duration_ms is not None:
        extra["duration_ms"] = round(float(duration_ms), 2)
    logger.log(level, message, extra=extra, exc_info=exc_info)


def uvicorn_log_config() -> dict[str, Any]:
    """Pass to uvicorn.run(log_config=...) so access logs are JSON / plain, not ANSI."""
    configure_logging()
    fmt = (os.getenv("LOG_FORMAT") or "json").strip().lower()
    use_json = fmt not in {"text", "plain", "human"}
    formatter_name = "json" if use_json else "plain"
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "context": {"()": "logger.logger.ContextFilter"},
        },
        "formatters": {
            "json": {"()": "logger.logger.JsonFormatter"},
            "plain": {
                "()": "logger.logger.PlainFormatter",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": formatter_name,
                "filters": ["context"],
                "stream": "ext://sys.stdout",
            },
            "access": {
                "class": "logging.StreamHandler",
                "formatter": formatter_name,
                "filters": ["context"],
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
            "sip": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }
