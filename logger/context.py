"""Request-scoped logging context (trace_id, ticker) via contextvars."""

from __future__ import annotations

from contextvars import ContextVar

_trace_id: ContextVar[str | None] = ContextVar("sip_trace_id", default=None)
_ticker: ContextVar[str | None] = ContextVar("sip_ticker", default=None)
_request_path: ContextVar[str | None] = ContextVar("sip_request_path", default=None)


def get_trace_id() -> str | None:
    return _trace_id.get()


def set_trace_id(value: str | None) -> None:
    _trace_id.set(value)


def get_ticker() -> str | None:
    return _ticker.get()


def set_ticker(value: str | None) -> None:
    _ticker.set(value)


def get_request_path() -> str | None:
    return _request_path.get()


def set_request_path(value: str | None) -> None:
    _request_path.set(value)


def clear_context() -> None:
    _trace_id.set(None)
    _ticker.set(None)
    _request_path.set(None)


def bind_context(
    *,
    trace_id: str | None = None,
    ticker: str | None = None,
    request_path: str | None = None,
) -> None:
    if trace_id is not None:
        set_trace_id(trace_id)
    if ticker is not None:
        set_ticker(ticker)
    if request_path is not None:
        set_request_path(request_path)
