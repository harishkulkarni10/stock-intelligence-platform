from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.ingestion import fetch_ohlcv, ingest_ticker, normalize_ticker
from src.data.preparation import prepare_sequences
from src.model.definition import LSTMForecaster
from src.model.evaluation import evaluate_forecast, select_champion
from src.model.saving import load_artifact, save_artifact
from src.model.training import train_model
from src.pipelines.data_pipeline import load_features
from src.utils import initialize_dirs, setup_mlflow


def _cfg() -> Config:
    return Config()


def parent_artifact_dir(cfg: Config | None = None) -> Path:
    config = cfg or _cfg()
    return Path(config.parent_dir)


def child_artifact_dir(ticker: str, cfg: Config | None = None) -> Path:
    config = cfg or _cfg()
    return Path(config.workdir) / normalize_ticker(ticker)


def _feature_frame(
    ticker: str,
    cfg: Config,
    *,
    persist: bool = True,
    source: str = "auto",
):
    """Resolve training data from the feature store, falling back to ingestion."""
    if source not in {"auto", "feature-store", "yfinance"}:
        raise ValueError("source must be 'auto', 'feature-store', or 'yfinance'")
    if source in {"auto", "feature-store"}:
        try:
            return load_features(ticker, path=cfg.feature_path, cfg=cfg)
        except (FileNotFoundError, KeyError):
            if source == "feature-store":
                raise

    kwargs = {
        "start": cfg.start_date,
        "period": None,
        "context_length": cfg.context_len,
        "prediction_length": cfg.pred_len,
    }
    if persist:
        return ingest_ticker(ticker, path=cfg.feature_path, **kwargs)
    return fetch_ohlcv(ticker, **kwargs)


def _prepare(frame, cfg: Config):
    return prepare_sequences(
        frame,
        context_length=cfg.context_len,
        prediction_length=cfg.pred_len,
        train_fraction=cfg.train_ratio,
        val_fraction=cfg.validation_ratio,
        feature_columns=tuple(cfg.features),
        target_column="Close",
    )


def _loaders(prepared, cfg: Config):
    return (
        DataLoader(prepared.train, batch_size=cfg.batch_size, shuffle=False),
        DataLoader(prepared.val, batch_size=cfg.batch_size, shuffle=False),
        DataLoader(prepared.test, batch_size=cfg.batch_size, shuffle=False),
    )


def _predict_loader(model: LSTMForecaster, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, _ in loader:
            preds = model(inputs.to(device)).detach().cpu().numpy()
            chunks.append(preds)
    return np.concatenate(chunks, axis=0)


def _test_anchors_and_returns(frame, prepared, cfg: Config):
    close = frame["Close"].to_numpy(dtype=np.float64)
    sample_count = len(frame) - cfg.context_len - cfg.pred_len + 1
    starts = np.arange(sample_count)
    target_starts = starts + cfg.context_len
    target_rows = target_starts[:, None] + np.arange(cfg.pred_len)
    anchors = close[target_starts - 1]
    returns = (close[target_rows] / anchors[:, None] - 1.0).astype(np.float32)
    test_mask = target_starts >= prepared.val_end
    return anchors[test_mask], returns[test_mask]


def _metadata(
    *,
    ticker: str,
    model: LSTMForecaster,
    cfg: Config,
    metrics: dict[str, Any],
    version: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "ticker": ticker,
        "model_type": type(model).__name__,
        "features": list(cfg.features),
        "target_mode": "cumulative_return",
        "transform": "simple",
        "lookback": cfg.context_len,
        "horizon": cfg.pred_len,
        "config": model.config,
        "metrics": metrics,
        "version": version,
    }
    if extra:
        payload.update(extra)
    return payload


def train_parent(
    *,
    epochs: int | None = None,
    persist_features: bool = True,
    source: str = "auto",
) -> dict[str, Any]:
    cfg = _cfg()
    initialize_dirs(cfg.workdir.parent)
    setup_mlflow()
    frame = _feature_frame(cfg.parent_ticker, cfg, persist=persist_features, source=source)
    prepared = _prepare(frame, cfg)
    train_loader, val_loader, test_loader = _loaders(prepared, cfg)
    model = LSTMForecaster(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        layers=cfg.num_layers,
        horizon=cfg.pred_len,
        dropout=cfg.dropout,
    )
    result = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs or cfg.parent_epochs,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
        device=cfg.device,
    )
    predicted = _predict_loader(model, test_loader, cfg.device)
    anchors, actual = _test_anchors_and_returns(frame, prepared, cfg)
    metrics = evaluate_forecast(predicted, actual, anchors)
    version = f"parent-{result['best_epoch']}"
    out_dir = parent_artifact_dir(cfg)
    save_artifact(
        out_dir,
        model,
        prepared.scaler,
        _metadata(
            ticker=cfg.parent_ticker,
            model=model,
            cfg=cfg,
            metrics=metrics,
            version=version,
            extra={"role": "parent", "best_epoch": result["best_epoch"]},
        ),
    )
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "ticker": cfg.parent_ticker,
                "best_epoch": result["best_epoch"],
                "best_loss": result["best_loss"],
                "history": result["history"],
                "metrics": metrics,
                "version": version,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return {
        "ticker": cfg.parent_ticker,
        "artifact_dir": str(out_dir),
        "best_epoch": result["best_epoch"],
        "best_loss": result["best_loss"],
        "history": result["history"],
        "metrics": metrics,
        "version": version,
    }


def train_child(
    ticker: str,
    *,
    epochs: int | None = None,
    persist_features: bool = True,
    child_improvement: float = 0.01,
    source: str = "auto",
) -> dict[str, Any]:
    cfg = _cfg()
    symbol = normalize_ticker(ticker)
    initialize_dirs(cfg.workdir.parent)
    parent_dir = parent_artifact_dir(cfg)
    if not (parent_dir / "model.pt").exists():
        raise FileNotFoundError(f"Parent model missing at {parent_dir}")

    parent_model, _, parent_meta = load_artifact(parent_dir, map_location=cfg.device)
    frame = _feature_frame(symbol, cfg, persist=persist_features, source=source)
    prepared = _prepare(frame, cfg)
    train_loader, val_loader, test_loader = _loaders(prepared, cfg)

    child = copy.deepcopy(parent_model)
    if cfg.transfer_strategy == "freeze":
        for name, param in child.named_parameters():
            if name.startswith("lstm."):
                param.requires_grad = False
        lr = cfg.learning_rate
    else:
        lr = cfg.fine_tune_lr

    result = train_model(
        child,
        train_loader,
        val_loader,
        epochs=epochs or cfg.child_epochs,
        learning_rate=lr,
        seed=cfg.seed,
        device=cfg.device,
    )

    parent_preds = _predict_loader(parent_model, test_loader, cfg.device)
    child_preds = _predict_loader(child, test_loader, cfg.device)
    anchors, actual = _test_anchors_and_returns(frame, prepared, cfg)
    parent_eval = evaluate_forecast(parent_preds, actual, anchors)
    child_eval = evaluate_forecast(child_preds, actual, anchors)
    champion = select_champion(
        {
            "persistence": parent_eval["persistence"],
            "parent": parent_eval["model"],
            "child": child_eval["model"],
        },
        minimum_child_improvement=child_improvement,
    )
    summary = {
        "ticker": symbol,
        "champion": champion,
        "parent_metrics": parent_eval,
        "child_metrics": child_eval,
        "parent_version": parent_meta.get("version"),
        "best_epoch": result["best_epoch"],
        "best_loss": result["best_loss"],
        "history": result["history"],
        "promoted": champion == "child",
    }
    if champion != "child":
        return summary

    out_dir = child_artifact_dir(symbol, cfg)
    version = f"child-{result['best_epoch']}"
    save_artifact(
        out_dir,
        child,
        prepared.scaler,
        _metadata(
            ticker=symbol,
            model=child,
            cfg=cfg,
            metrics=child_eval,
            version=version,
            extra={
                "role": "child",
                "parent_version": parent_meta.get("version"),
                "transfer_strategy": cfg.transfer_strategy,
                "best_epoch": result["best_epoch"],
            },
        ),
    )
    summary["artifact_dir"] = str(out_dir)
    summary["version"] = version
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train parent or child forecast models")
    parser.add_argument("mode", choices=["parent", "child"])
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--source", choices=["auto", "feature-store", "yfinance"], default="auto"
    )
    args = parser.parse_args(argv)
    if args.mode == "parent":
        summary = train_parent(epochs=args.epochs, source=args.source)
        print(json.dumps(summary, indent=2, default=str))
        return
    if not args.ticker:
        raise SystemExit("--ticker is required for child training")
    summary = train_child(args.ticker, epochs=args.epochs, source=args.source)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
