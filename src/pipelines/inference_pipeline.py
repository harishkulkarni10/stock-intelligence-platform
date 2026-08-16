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


def _latest_window(frame, feature_columns: list[str], lookback: int) -> np.ndarray:
    if len(frame) < lookback:
        raise ValueError(f"Need at least {lookback} rows, got {len(frame)}")
    return frame.iloc[-lookback:][feature_columns].to_numpy(dtype=np.float32)


def _predict_artifact(path: Path, ticker: str, horizon: int | None = None) -> dict[str, Any]:
    cfg = _cfg()
    model, scaler, meta = load_artifact(path, map_location=cfg.device)
    feature_columns = list(meta.get("features") or cfg.features)
    lookback = int(meta.get("lookback") or cfg.context_len)
    model_horizon = int(meta.get("horizon") or model.config["horizon"])
    requested = horizon or model_horizon
    if requested > model_horizon:
        raise ValueError(f"Requested horizon {requested} exceeds model horizon {model_horizon}")

    frame = fetch_ohlcv(
        ticker,
        start=cfg.start_date,
        period=None,
        context_length=lookback,
        prediction_length=1,
    )
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


def predict_parent(ticker: str | None = None, horizon: int | None = None) -> dict[str, Any]:
    cfg = _cfg()
    symbol = normalize_ticker(ticker or cfg.parent_ticker)
    return _predict_artifact(_resolve_dir("parent"), symbol, horizon)


def predict_child(ticker: str, horizon: int | None = None) -> dict[str, Any]:
    symbol = normalize_ticker(ticker)
    return _predict_artifact(_resolve_dir("child", symbol), symbol, horizon)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run forecast inference")
    parser.add_argument("mode", choices=["parent", "child"])
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    args = parser.parse_args(argv)
    if args.mode == "parent":
        print(json.dumps(predict_parent(args.ticker, args.horizon), indent=2, default=str))
        return
    if not args.ticker:
        raise SystemExit("--ticker is required for child prediction")
    print(json.dumps(predict_child(args.ticker, args.horizon), indent=2, default=str))


if __name__ == "__main__":
    main()
