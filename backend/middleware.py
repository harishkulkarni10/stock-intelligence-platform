"""HTTP middleware: assign trace_id and bind logging context."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from logger.context import bind_context, clear_context
from logger.logger import get_logger, log_event

logger = get_logger("sip.http")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming = request.headers.get("x-request-id") or request.headers.get(
            "x-correlation-id"
        )
        trace_id = (incoming or "").strip() or f"req-{uuid.uuid4().hex[:12]}"
        bind_context(trace_id=trace_id, request_path=request.url.path)
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = trace_id
            return response
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            # Skip noisy static asset spam at DEBUG-ish volume.
            if request.url.path not in {"/styles.css", "/app.js", "/favicon.ico"}:
                log_event(
                    logger,
                    "http_request",
                    step="http",
                    status="ok" if status_code < 400 else "error",
                    duration_ms=duration_ms,
                    data={
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": status_code,
                    },
                )
            clear_context()
