from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import redis
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

redis_client: redis.Redis | None = None
# V1 background work is process-local. Keep API_WORKERS=1 until this is replaced
# by a durable queue, and serialize model training/inference in this process.
executor = ThreadPoolExecutor(max_workers=1)

registry = CollectorRegistry()
REDIS_UP = Gauge("redis_up", "Redis connectivity", registry=registry)
SYSTEM_CPU = Gauge("system_cpu_percent", "CPU percent", registry=registry)
PREDICTION_TOTAL = Counter(
    "prediction_total", "Predictions served", ["type"], registry=registry
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency", ["type"], registry=registry
)
CACHE_HIT = Counter("cache_hit_total", "Cache hits", ["key"], registry=registry)
CACHE_MISS = Counter("cache_miss_total", "Cache misses", ["key"], registry=registry)


def get_redis() -> redis.Redis | None:
    return redis_client
