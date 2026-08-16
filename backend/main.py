from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import redis
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

import backend.state as app_state
from backend.api import router
from backend.schemas import HealthResponse
from backend.state import REDIS_UP, registry

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]


def _redis_url() -> str:
    if url := os.getenv("REDIS_URL"):
        return url
    host = os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_PORT", "6379")
    return f"redis://{host}:{port}/0"


@asynccontextmanager
async def lifespan(_: FastAPI):
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    try:
        client = redis.Redis.from_url(_redis_url(), decode_responses=True)
        client.ping()
        app_state.redis_client = client
        REDIS_UP.set(1)
    except redis.RedisError:
        app_state.redis_client = None
        REDIS_UP.set(0)
    yield
    if app_state.redis_client is not None:
        try:
            app_state.redis_client.close()
        except redis.RedisError:
            pass
        app_state.redis_client = None


app = FastAPI(title="Stock Intelligence Platform", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    redis_ok = False
    client = app_state.get_redis()
    if client is not None:
        try:
            redis_ok = bool(client.ping())
        except redis.RedisError:
            redis_ok = False

    parent_present = (ROOT / "outputs" / "parent" / "model.pt").exists()
    features = ROOT / "feature_store" / "data" / "features.parquet"
    details = {
        "redis": redis_ok,
        "parent_model": parent_present,
        "features": features.exists(),
    }
    require_redis = os.getenv("REQUIRE_REDIS", "false").lower() == "true"
    require_models = os.getenv("REQUIRE_MODELS", "false").lower() == "true"
    if (require_redis and not redis_ok) or (
        require_models and not (parent_present and features.exists())
    ):
        raise HTTPException(status_code=503, detail={"status": "not_ready", **details})
    return HealthResponse(status="ready", details=details)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


def run() -> None:
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
        workers=int(os.getenv("API_WORKERS", "1")),
    )


if __name__ == "__main__":
    run()
