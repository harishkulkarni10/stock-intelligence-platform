"""Ticker-keyed report cache with TTL (Redis). Fail-open if Redis is down."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from backend import state as app_state


class ReportCache:
    """Stores analyze DTOs by ticker for a short TTL. Not document RAG."""

    def __init__(self, *, ttl_seconds: int | None = None, prefix: str = "analyze-report"):
        self.ttl_seconds = ttl_seconds or int(os.getenv("ANALYZE_CACHE_TTL_SECONDS", "3600"))
        self.prefix = prefix

    def _key(self, ticker: str) -> str:
        return f"{self.prefix}:{ticker.upper()}"

    def get(self, ticker: str) -> dict[str, Any] | None:
        client = app_state.get_redis()
        if client is None:
            return None
        try:
            raw = client.get(self._key(ticker))
            if not raw:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode()
            payload = json.loads(raw)
            created = int(payload.get("cached_at_ts", 0))
            if created and time.time() - created > self.ttl_seconds:
                client.delete(self._key(ticker))
                return None
            return payload.get("result")
        except Exception:
            return None

    def set(self, ticker: str, result: dict[str, Any]) -> None:
        client = app_state.get_redis()
        if client is None:
            return
        try:
            envelope = {"cached_at_ts": int(time.time()), "result": result}
            client.setex(self._key(ticker), self.ttl_seconds, json.dumps(envelope, default=str))
        except Exception:
            return


# Backward-compatible name used in docs / Karan layout.
SemanticCache = ReportCache
