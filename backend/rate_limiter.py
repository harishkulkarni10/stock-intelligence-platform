"""Small, fail-open Redis fixed-window rate limiter."""

from __future__ import annotations

from backend import state


class FixedWindowRateLimiter:
    """Limit a key to ``limit`` calls in a fixed Redis time window."""

    def __init__(self, prefix: str = "rate") -> None:
        self.prefix = prefix

    def is_allowed(self, key: str, limit: int, window_seconds: int) -> bool:
        client = state.get_redis()
        if client is None:
            return True

        redis_key = f"{self.prefix}:{key}"
        try:
            count = int(client.incr(redis_key))
            if count == 1:
                client.expire(redis_key, window_seconds)
            return count <= limit
        except Exception:  # noqa: BLE001 - Redis must fail open
            return True
