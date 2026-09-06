from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.data.ingestion import fetch_ohlcv, normalize_ticker
from src.inference import decode_forecast
from src.model.saving import load_artifact
from src.pipelines.training_pipeline import child_artifact_dir, parent_artifact_dir


def _cfg() -> Config:
    return Config()


def _resolve_dir(model_type: str, ticker: str | None = None) -> Path:
    cfg = _cfg()
    if model_type == "parent":
        path = parent_artifact_dir(cfg)
    else:
        if not ticker:
            raise ValueError("ticker is required for child prediction")
        path = child_artifact_dir(ticker, cfg)
    if not (path / "model.pt").exists():
        raise FileNotFoundError(f"Missing model artifact at {path}")
    return path


def _child_model_exists(ticker: str) -> bool:
    return (child_artifact_dir(ticker, _cfg()) / "model.pt").exists()
    

def _load_train_summary(ticker: str) -> dict[str, Any] | None:
    path = child_artifact_dir(ticker, _cfg()) / "train_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_serving_source(ticker: str) -> str:
    """Pick child, parent, or persistence using champion results when available."""
    symbol = normalize_ticker(ticker)
    if _child_model_exists(symbol):
        return "child"
    summary = _load_train_summary(symbol)
    if summary is not None:
        return str(summary.get("champion", "parent"))
    if (parent_artifact_dir(_cfg()) / "model.pt").exists():
        return "parent"
    return "persistence"


def _eval_metadata(ticker: str, source: str) -> dict[str, Any]:
    symbol = normalize_ticker(ticker)
    summary = _load_train_summary(symbol)
    if summary is None and source == "parent":
        parent_summary_path = parent_artifact_dir(_cfg()) / "train_summary.json"
        if parent_summary_path.exists():
            summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))
    if summary is None:
        return {"serving_source": source, "champion": source, "beats_persistence": None}

    metrics_key = {
        "child": "child_metrics",
        "parent": "parent_metrics",
        "persistence": "parent_metrics",
    }.get(source, "parent_metrics")
    metrics_block = summary.get(metrics_key) or summary.get("metrics") or {}
    model_metrics = metrics_block.get("model", metrics_block) if isinstance(metrics_block, dict) else {}
    persistence_metrics = metrics_block.get("persistence", {}) if isinstance(metrics_block, dict) else {}
    return {
        "serving_source": source,
        "champion": summary.get("champion", source),
        "beats_persistence": summary.get("beats_persistence"),
        "test_mae": model_metrics.get("mae") if source != "persistence" else persistence_metrics.get("mae"),
        "test_persistence_mae": persistence_metrics.get("mae"),
        "promoted": summary.get("promoted"),
        "parent_version": summary.get("parent_version") or summary.get("version"),
    }


def _latest_window(frame, feature_columns: list[str], lookback: int) -> np.ndarray:
    if len(frame) < lookback:
        raise ValueError(f"Need at least {lookback} rows, got {len(frame)}")
    return frame.iloc[-lookback:][feature_columns].to_numpy(dtype=np.float32)


def _live_frame(ticker: str, lookback: int) -> pd.DataFrame:
    cfg = _cfg()
    return fetch_ohlcv(
        ticker,
        start=cfg.start_date,
        period=None,
        context_length=lookback,
        prediction_length=1,
    )


def _predict_artifact(path: Path, ticker: str, horizon: int | None = None) -> dict[str, Any]:
    cfg = _cfg()
    model, scaler, meta = load_artifact(path, map_location=cfg.device)
    feature_columns = list(meta.get("features") or cfg.features)
    lookback = int(meta.get("lookback") or cfg.context_len)
    model_horizon = int(meta.get("horizon") or model.config["horizon"])
    requested = horizon or model_horizon
    if requested > model_horizon:
        raise ValueError(f"Requested horizon {requested} exceeds model horizon {model_horizon}")

    frame = _live_frame(ticker, lookback)
    window = _latest_window(frame, feature_columns, lookback)
    scaled = scaler.transform(window).astype(np.float32)
    tensor = torch.from_numpy(scaled).unsqueeze(0).to(cfg.device)
    model.eval()
    with torch.no_grad():
        raw = model(tensor).detach().cpu().numpy()[0][:requested]

    last_close = float(frame.iloc[-1]["Close"])
    last_date = pd.Timestamp(frame.iloc[-1]["date"])
    series = decode_forecast(
        raw, last_close, last_date, transform=meta.get("transform", "simple")
    )
    history = [
        {
            "date": pd.Timestamp(row_date).strftime("%Y-%m-%d"),
            "close": float(close),
        }
        for row_date, close in zip(
            frame.tail(30)["date"].tolist(),
            frame.tail(30)["Close"].tolist(),
            strict=True,
        )
    ]
    predictions = [
        {
            "step": index,
            "date": ts.strftime("%Y-%m-%d"),
            "close": float(price),
            "value": float(price),
        }
        for index, (ts, price) in enumerate(series.items(), start=1)
    ]
    return {
        "ticker": normalize_ticker(ticker),
        "horizon": requested,
        "predictions": predictions,
        "history": history,
        "last_close": last_close,
        "last_date": last_date.strftime("%Y-%m-%d"),
        "model_version": meta.get("version"),
        "model_type": meta.get("role") or meta.get("model_type"),
        "target_mode": meta.get("target_mode", "cumulative_return"),
        "artifact_dir": str(path),
    }


def predict_persistence(ticker: str, horizon: int | None = None) -> dict[str, Any]:
    """Flat-price baseline: zero cumulative return over the requested horizon."""
    cfg = _cfg()
    symbol = normalize_ticker(ticker)
    requested = horizon or cfg.pred_len
    frame = _live_frame(symbol, cfg.context_len)
    last_close = float(frame.iloc[-1]["Close"])
    last_date = pd.Timestamp(frame.iloc[-1]["date"])
    series = decode_forecast(
        np.zeros(requested, dtype=float),
        last_close,
        last_date,
        transform="simple",
    )
    history = [
        {
            "date": pd.Timestamp(row_date).strftime("%Y-%m-%d"),
            "close": float(close),
        }
        for row_date, close in zip(
            frame.tail(30)["date"].tolist(),
            frame.tail(30)["Close"].tolist(),
            strict=True,
        )
    ]
    predictions = [
        {
            "step": index,
            "date": ts.strftime("%Y-%m-%d"),
            "close": float(price),
            "value": float(price),
        }
        for index, (ts, price) in enumerate(series.items(), start=1)
    ]
    return {
        "ticker": symbol,
        "horizon": requested,
        "predictions": predictions,
        "history": history,
        "last_close": last_close,
        "last_date": last_date.strftime("%Y-%m-%d"),
        "model_version": "persistence",
        "model_type": "persistence",
        "target_mode": "cumulative_return",
        "artifact_dir": None,
    }


def predict_parent(ticker: str | None = None, horizon: int | None = None) -> dict[str, Any]:
    cfg = _cfg()
    symbol = normalize_ticker(ticker or cfg.parent_ticker)
    return _predict_artifact(_resolve_dir("parent"), symbol, horizon)


def predict_child(ticker: str, horizon: int | None = None) -> dict[str, Any]:
    symbol = normalize_ticker(ticker)
    return _predict_artifact(_resolve_dir("child", symbol), symbol, horizon)


def predict_best(ticker: str, horizon: int | None = None) -> dict[str, Any]:
    """Serve the evaluation winner: child artifact, parent, or persistence."""
    symbol = normalize_ticker(ticker)
    source = resolve_serving_source(symbol)
    if source == "child":
        payload = predict_child(symbol, horizon)
    elif source == "persistence":
        payload = predict_persistence(symbol, horizon)
    else:
        payload = predict_parent(symbol, horizon)
    payload["model_source"] = source
    payload["evaluation"] = _eval_metadata(symbol, source)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run forecast inference")
    parser.add_argument("mode", choices=["parent", "child", "best", "persistence"])
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    args = parser.parse_args(argv)
    if args.mode == "best":
        if not args.ticker:
            raise SystemExit("--ticker is required for best prediction")
        print(json.dumps(predict_best(args.ticker, args.horizon), indent=2, default=str))
        return
    if args.mode == "persistence":
        if not args.ticker:
            raise SystemExit("--ticker is required for persistence prediction")
        print(json.dumps(predict_persistence(args.ticker, args.horizon), indent=2, default=str))
        return
    if args.mode == "parent":
        print(json.dumps(predict_parent(args.ticker, args.horizon), indent=2, default=str))
        return
    if not args.ticker:
        raise SystemExit("--ticker is required for child prediction")
    print(json.dumps(predict_child(args.ticker, args.horizon), indent=2, default=str))


if __name__ == "__main__":
    main()
