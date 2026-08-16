"""Process-local task orchestration and fail-open Redis persistence."""

from __future__ import annotations

import importlib
import json
import os
import threading
import uuid
from collections.abc import Callable
from typing import Any

from backend import state

TASK_TTL_SECONDS = int(os.getenv("TASK_TTL_SECONDS", "86400"))
CACHE_TTL_SECONDS = int(os.getenv("PREDICTION_CACHE_TTL_SECONDS", "3600"))

_local_tasks: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _json(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def _set_task(task: dict[str, Any]) -> None:
    with _lock:
        _local_tasks[task["task_id"]] = task.copy()
    client = state.get_redis()
    if client is not None:
        try:
            client.setex(f"task:{task['task_id']}", TASK_TTL_SECONDS, _json(task))
        except Exception:  # noqa: BLE001 - Redis persistence is fail-open
            return


def get_task_status(task_id: str) -> dict[str, Any] | None:
    client = state.get_redis()
    if client is not None:
        try:
            raw = client.get(f"task:{task_id}")
            if raw:
                if isinstance(raw, bytes):
                    raw = raw.decode()
                return json.loads(raw)
        except Exception:  # noqa: BLE001 - fall back to process-local status
            raw = None
    with _lock:
        task = _local_tasks.get(task_id)
        return task.copy() if task else None


def submit_task(task_type: str, function: Callable[..., Any], *args: Any) -> str:
    task_id = uuid.uuid4().hex
    task = {"task_id": task_id, "task_type": task_type, "status": "queued"}
    _set_task(task)

    def run() -> None:
        task["status"] = "running"
        _set_task(task)
        try:
            task["result"] = function(*args)
            task["status"] = "completed"
        except Exception as exc:  # noqa: BLE001 - task failures become status data
            task["status"] = "failed"
            task["error"] = str(exc)
        _set_task(task)

    state.executor.submit(run)
    return task_id


def _pipeline_function(module_name: str, function_name: str) -> Callable[..., Any]:
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise TypeError(f"{module_name}.{function_name} is not implemented")
    return function


def train_parent() -> Any:
    return _pipeline_function(
        "src.pipelines.training_pipeline", "train_parent"
    )()


def train_child(ticker: str) -> Any:
    return _pipeline_function(
        "src.pipelines.training_pipeline", "train_child"
    )(ticker)


def train_child_after_parent(ticker: str) -> Any:
    train_parent()
    return train_child(ticker)


def predict_parent(ticker: str, horizon: int) -> Any:
    return _pipeline_function(
        "src.pipelines.inference_pipeline", "predict_parent"
    )(ticker, horizon)


def predict_child(ticker: str, horizon: int) -> Any:
    return _pipeline_function(
        "src.pipelines.inference_pipeline", "predict_child"
    )(ticker, horizon)


def prediction_cache_key(
    prediction_type: str, ticker: str, horizon: int, model_version: str
) -> str:
    return f"prediction:{prediction_type}:{ticker}:{horizon}:{model_version}"


def get_cached_prediction(
    prediction_type: str, ticker: str, horizon: int
) -> dict[str, Any] | None:
    client = state.get_redis()
    if client is None:
        return None
    index_key = f"prediction-version:{prediction_type}:{ticker}:{horizon}"
    try:
        version = client.get(index_key)
        if not version:
            return None
        if isinstance(version, bytes):
            version = version.decode()
        raw = client.get(prediction_cache_key(prediction_type, ticker, horizon, version))
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        return json.loads(raw)
    except Exception:  # noqa: BLE001 - prediction cache is fail-open
        return None


def cache_prediction(prediction_type: str, result: dict[str, Any]) -> None:
    client = state.get_redis()
    if client is None:
        return
    ticker = str(result["ticker"])
    horizon = int(result["horizon"])
    version = str(result.get("model_version") or "unknown")
    index_key = f"prediction-version:{prediction_type}:{ticker}:{horizon}"
    cache_key = prediction_cache_key(prediction_type, ticker, horizon, version)
    try:
        client.setex(cache_key, CACHE_TTL_SECONDS, _json(result))
        client.setex(index_key, CACHE_TTL_SECONDS, version)
    except Exception:  # noqa: BLE001 - prediction cache is fail-open
        return
