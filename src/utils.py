from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

from src.config import Config


def initialize_dirs(root: Path | None = None) -> None:
    base = root or Config().workdir.parent
    for relative in ("outputs", "outputs/parent", "logs", "feature_store/data"):
        (base / relative).mkdir(parents=True, exist_ok=True)


def setup_mlflow(experiment: str = "stock-intelligence-forecasting") -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI", Config().mlflow_tracking_uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


@contextmanager
def mlflow_training_run(run_name: str, *, experiment: str = "stock-intelligence-forecasting") -> Iterator[Any]:
    """Start an MLflow run after binding the configured tracking URI."""
    setup_mlflow(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_training_params(**params: Any) -> None:
    for key, value in params.items():
        mlflow.log_param(key, value if not isinstance(value, (list, tuple)) else ",".join(map(str, value)))


def log_eval_metrics(prefix: str, metrics: dict[str, Any]) -> None:
    """Log nested evaluation dicts as flat MLflow metrics (e.g. model_mae)."""
    for group, values in metrics.items():
        if isinstance(values, dict):
            for name, value in values.items():
                mlflow.log_metric(f"{prefix}_{group}_{name}", float(value))
        else:
            mlflow.log_metric(f"{prefix}_{group}", float(values))


def log_training_history(history: dict[str, list[float]]) -> None:
    for epoch, train_loss in enumerate(history.get("train_loss", []), start=1):
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric(
            "validation_loss",
            history["validation_loss"][epoch - 1],
            step=epoch,
        )
        mlflow.log_metric("learning_rate", history["learning_rate"][epoch - 1], step=epoch)


def log_artifact_dir(directory: str | Path, artifact_path: str = "artifacts") -> None:
    path = Path(directory)
    if not path.exists():
        return
    for name in ("model.pt", "scaler.joblib", "meta.json", "train_summary.json"):
        candidate = path / name
        if candidate.exists():
            mlflow.log_artifact(str(candidate), artifact_path)
