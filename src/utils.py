from __future__ import annotations

import os
from pathlib import Path

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
