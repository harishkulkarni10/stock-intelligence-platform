from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Response, status

from backend import state, tasks
from backend.rate_limiter import FixedWindowRateLimiter
from backend.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    PredictionPoint,
    PredictionResult,
    PredictRequest,
    TaskAccepted,
    TaskStatus,
    TrainChildRequest,
)

router = APIRouter()
ROOT = Path(__file__).resolve().parents[1]
rate_limiter = FixedWindowRateLimiter()


@router.get("/")
def root() -> dict:
    return {
        "project": "Stock Intelligence Platform",
        "version": "0.1.0",
        "status": "stage1",
        "endpoints": {
            "health": "GET /health",
            "ready": "GET /ready",
            "train_parent": "POST /train-parent",
            "train_child": "POST /train-child",
            "predict_parent": "POST /predict-parent",
            "predict_child": "POST /predict-child",
            "analyze": "POST /analyze",
            "status": "GET /status/{task_id}",
            "docs": "GET /docs",
        },
    }


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", details={"service": "api"})


def _parent_model_exists() -> bool:
    return (ROOT / "outputs" / "parent" / "model.pt").exists()


def _child_model_exists(ticker: str) -> bool:
    return (ROOT / "outputs" / ticker.upper() / "model.pt").exists()


def _limit(scope: str, ticker: str = "") -> None:
    limit = int(os.getenv("API_RATE_LIMIT", "120"))
    window = int(os.getenv("API_RATE_WINDOW_SECONDS", "60"))
    if not rate_limiter.is_allowed(f"{scope}:{ticker}", limit, window):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


def _accepted(task_id: str) -> dict[str, str]:
    return {"status": "queued", "task_id": task_id}


def _normalize_prediction(raw: Any, ticker: str, horizon: int) -> dict[str, Any]:
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    payload = dict(raw) if isinstance(raw, dict) else {"predictions": raw}
    values = payload.get("predictions", [])
    points: list[PredictionPoint] = []
    for index, item in enumerate(values, start=1):
        if isinstance(item, dict):
            value = item.get(
                "value",
                item.get("predicted_close", item.get("prediction", item.get("close"))),
            )
            if value is None:
                raise ValueError("Prediction point has no numeric value")
            points.append(
                PredictionPoint(
                    step=int(item.get("step", index)),
                    value=float(value),
                    timestamp=item.get("timestamp", item.get("date")),
                )
            )
        else:
            points.append(PredictionPoint(step=index, value=float(item)))
    result = PredictionResult(
        ticker=str(payload.get("ticker", ticker)).upper(),
        horizon=int(payload.get("horizon", horizon)),
        predictions=points,
        model_version=payload.get("model_version"),
    ).model_dump()
    for key, value in payload.items():
        if key not in result:
            result[key] = value
    return result


@router.post("/train-parent", status_code=status.HTTP_202_ACCEPTED)
def train_parent() -> dict[str, str]:
    _limit("train", "parent")
    return _accepted(tasks.submit_task("train-parent", tasks.train_parent))


@router.post("/train-child", status_code=status.HTTP_202_ACCEPTED)
def train_child(request: TrainChildRequest) -> dict[str, str]:
    _limit("train", request.ticker)
    function = tasks.train_child if _parent_model_exists() else tasks.train_child_after_parent
    return _accepted(tasks.submit_task("train-child", function, request.ticker))


async def _predict(
    prediction_type: str, request: PredictRequest, function: Any
) -> PredictionResult:
    cached = tasks.get_cached_prediction(
        prediction_type, request.ticker, request.horizon or 5
    )
    if cached is not None:
        return PredictionResult.model_validate(cached)
    horizon = request.horizon or 5
    raw = await asyncio.get_running_loop().run_in_executor(
        state.executor, function, request.ticker, horizon
    )
    result = _normalize_prediction(raw, request.ticker, horizon)
    tasks.cache_prediction(prediction_type, result)
    return PredictionResult.model_validate(result)


@router.post("/predict-parent", response_model=PredictionResult | TaskAccepted)
async def predict_parent(
    request: PredictRequest, response: Response
) -> PredictionResult | dict[str, str]:
    _limit("predict-parent", request.ticker)
    if not _parent_model_exists():
        response.status_code = status.HTTP_202_ACCEPTED
        return _accepted(tasks.submit_task("train-parent", tasks.train_parent))
    return await _predict("parent", request, tasks.predict_parent)


@router.post("/predict-child", response_model=PredictionResult | TaskAccepted)
async def predict_child(
    request: PredictRequest, response: Response
) -> PredictionResult | dict[str, str]:
    _limit("predict-child", request.ticker)
    if not _child_model_exists(request.ticker):
        function = (
            tasks.train_child if _parent_model_exists() else tasks.train_child_after_parent
        )
        response.status_code = status.HTTP_202_ACCEPTED
        return _accepted(tasks.submit_task("train-child", function, request.ticker))
    return await _predict("child", request, tasks.predict_child)


@router.get("/status/{task_id}", response_model=TaskStatus)
def task_status(task_id: str) -> TaskStatus:
    task = tasks.get_task_status(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatus.model_validate(task)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    _limit("analyze", request.ticker)

    def run() -> dict[str, Any]:
        from src.agents.graph import analyze_stock

        return analyze_stock(request.ticker, request.thread_id)

    result = await asyncio.get_running_loop().run_in_executor(state.executor, run)
    if result.get("status") not in {"ok", "missing_model", "error", "training"}:
        result = {**result, "status": result.get("status") or "error"}
    return AnalyzeResponse.model_validate(result)
