from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    context_len: int = 60
    pred_len: int = 5
    features: list[str] = field(
        default_factory=lambda: ["Open", "High", "Low", "Close", "Volume", "RSI14", "MACD"]
    )
    batch_size: int = 32
    parent_ticker: str = "^GSPC"
    child_tickers: list[str] = field(default_factory=lambda: ["NVDA", "AAPL", "MSFT"])
    start_date: str = "2004-08-19"
    parent_epochs: int = 20
    child_epochs: int = 10
    transfer_strategy: str = "freeze"
    learning_rate: float = 1e-3
    fine_tune_lr: float = 1e-4
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    seed: int = 42
    parent_dir: Path = ROOT_DIR / "outputs" / "parent"
    workdir: Path = ROOT_DIR / "outputs"
    feature_path: Path = ROOT_DIR / "feature_store" / "data" / "features.parquet"
    mlflow_tracking_uri: str = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"

    @property
    def input_size(self) -> int:
        return len(self.features)
