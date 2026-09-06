from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import mlflow

from src.config import Config
from src.data.ingestion import fetch_ohlcv, ingest_ticker, normalize_ticker
from src.data.preparation import prepare_sequences
from src.model.definition import LSTMForecaster
from src.model.evaluation import evaluate_forecast, select_champion
from src.model.saving import load_artifact, save_artifact
from src.model.training import train_model
from src.pipelines.data_pipeline import load_features
from src.utils import (
    initialize_dirs,
    log_artifact_dir,
    log_eval_metrics,
    log_training_history,
    log_training_params,
    mlflow_training_run,
)


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


def _write_train_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def train_parent(
    *,
    epochs: int | None = None,
    persist_features: bool = True,
    source: str = "auto",
) -> dict[str, Any]:
    cfg = _cfg()
    initialize_dirs(cfg.workdir.parent)
    symbol = cfg.parent_ticker
    with mlflow_training_run(f"parent_{symbol}") as run:
        log_training_params(
            role="parent",
            ticker=symbol,
            context_len=cfg.context_len,
            pred_len=cfg.pred_len,
            features=cfg.features,
            batch_size=cfg.batch_size,
            epochs=epochs or cfg.parent_epochs,
            learning_rate=cfg.learning_rate,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            transfer_strategy=cfg.transfer_strategy,
            data_source=source,
        )
        frame = _feature_frame(symbol, cfg, persist=persist_features, source=source)
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
        log_training_history(result["history"])
        predicted = _predict_loader(model, test_loader, cfg.device)
        anchors, actual = _test_anchors_and_returns(frame, prepared, cfg)
        metrics = evaluate_forecast(predicted, actual, anchors)
        champion = select_champion(
            {"persistence": metrics["persistence"], "parent": metrics["model"]}
        )
        version = f"parent-{result['best_epoch']}"
        out_dir = parent_artifact_dir(cfg)
        save_artifact(
            out_dir,
            model,
            prepared.scaler,
            _metadata(
                ticker=symbol,
                model=model,
                cfg=cfg,
                metrics=metrics,
                version=version,
                extra={"role": "parent", "best_epoch": result["best_epoch"]},
            ),
        )
        summary = {
            "ticker": symbol,
            "role": "parent",
            "champion": champion,
            "beats_persistence": champion != "persistence",
            "best_epoch": result["best_epoch"],
            "best_loss": result["best_loss"],
            "history": result["history"],
            "metrics": metrics,
            "version": version,
            "run_id": run.info.run_id,
        }
        _write_train_summary(out_dir / "train_summary.json", summary)
        log_eval_metrics("test", metrics)
        mlflow.log_metric("best_epoch", float(result["best_epoch"]))
        mlflow.log_metric("best_loss", float(result["best_loss"]))
        mlflow.log_param("champion", champion)
        mlflow.log_metric("beats_persistence", 1.0 if champion != "persistence" else 0.0)
        log_artifact_dir(out_dir)
        return {
            "ticker": symbol,
            "artifact_dir": str(out_dir),
            "champion": champion,
            "beats_persistence": champion != "persistence",
            "best_epoch": result["best_epoch"],
            "best_loss": result["best_loss"],
            "history": result["history"],
            "metrics": metrics,
            "version": version,
            "run_id": run.info.run_id,
        }


def train_child(
    ticker: str,
    *,
    epochs: int | None = None,
    persist_features: bool = True,
    child_improvement: float = 0.01,
    transfer_strategy: str | None = None,
    source: str = "auto",
) -> dict[str, Any]:
    cfg = _cfg()
    symbol = normalize_ticker(ticker)
    initialize_dirs(cfg.workdir.parent)
    parent_dir = parent_artifact_dir(cfg)
    if not (parent_dir / "model.pt").exists():
        raise FileNotFoundError(f"Parent model missing at {parent_dir}")

    with mlflow_training_run(f"child_{symbol}") as run:
        log_training_params(
            role="child",
            ticker=symbol,
            context_len=cfg.context_len,
            pred_len=cfg.pred_len,
            features=cfg.features,
            batch_size=cfg.batch_size,
            epochs=epochs or cfg.child_epochs,
            transfer_strategy=transfer_strategy or cfg.transfer_strategy,
            child_improvement=child_improvement,
            data_source=source,
        )
        parent_model, _, parent_meta = load_artifact(parent_dir, map_location=cfg.device)
        mlflow.log_param("parent_version", parent_meta.get("version", "unknown"))
        frame = _feature_frame(symbol, cfg, persist=persist_features, source=source)
        prepared = _prepare(frame, cfg)
        train_loader, val_loader, test_loader = _loaders(prepared, cfg)

        strategy = transfer_strategy or cfg.transfer_strategy
        child = copy.deepcopy(parent_model)
        if strategy == "freeze":
            for name, param in child.named_parameters():
                if name.startswith("lstm."):
                    param.requires_grad = False
            lr = cfg.learning_rate
        else:
            lr = cfg.fine_tune_lr
        mlflow.log_param("learning_rate", lr)

        result = train_model(
            child,
            train_loader,
            val_loader,
            epochs=epochs or cfg.child_epochs,
            learning_rate=lr,
            seed=cfg.seed,
            device=cfg.device,
        )
        log_training_history(result["history"])

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
        log_eval_metrics("test_parent", parent_eval)
        log_eval_metrics("test_child", child_eval)
        log_eval_metrics("test_persistence", {"persistence": parent_eval["persistence"]})
        mlflow.log_metric("best_epoch", float(result["best_epoch"]))
        mlflow.log_metric("best_loss", float(result["best_loss"]))
        mlflow.log_param("champion", champion)
        mlflow.log_metric("promoted", 1.0 if champion == "child" else 0.0)
        mlflow.log_metric("beats_persistence", 1.0 if champion != "persistence" else 0.0)

        out_dir = child_artifact_dir(symbol, cfg)
        summary = {
            "ticker": symbol,
            "role": "child",
            "champion": champion,
            "beats_persistence": champion != "persistence",
            "transfer_strategy": strategy,
            "child_improvement": child_improvement,
            "parent_metrics": parent_eval,
            "child_metrics": child_eval,
            "parent_version": parent_meta.get("version"),
            "best_epoch": result["best_epoch"],
            "best_loss": result["best_loss"],
            "history": result["history"],
            "promoted": champion == "child",
            "run_id": run.info.run_id,
        }
        if champion == "child":
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
                        "transfer_strategy": strategy,
                        "best_epoch": result["best_epoch"],
                    },
                ),
            )
            summary["artifact_dir"] = str(out_dir)
            summary["version"] = version
            log_artifact_dir(out_dir)
        _write_train_summary(out_dir / "train_summary.json", summary)
        return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train parent or child forecast models")
    parser.add_argument("mode", choices=["parent", "child"])
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--transfer-strategy",
        choices=["freeze", "full"],
        default=None,
        help="Child fine-tuning mode: freeze LSTM layers or train all weights",
    )
    parser.add_argument(
        "--child-improvement",
        type=float,
        default=0.01,
        help="Minimum relative MAE gain required to promote a child model",
    )
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
    summary = train_child(
        args.ticker,
        epochs=args.epochs,
        source=args.source,
        transfer_strategy=args.transfer_strategy,
        child_improvement=args.child_improvement,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
