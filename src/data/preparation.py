from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

DEFAULT_FEATURES = ("Open", "High", "Low", "Close", "Volume", "RSI14", "MACD")


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.inputs = torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32))
        self.targets = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


@dataclass(frozen=True)
class PreparedSequences:
    train: SequenceDataset
    val: SequenceDataset
    test: SequenceDataset
    scaler: StandardScaler
    feature_columns: tuple[str, ...]
    train_end: int
    val_end: int


def _validate_input(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    required = {"date", *features}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input data is empty")

    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("Dates must be valid, strictly increasing, and unique")
    values = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Features must be numeric and finite")
    return values


def prepare_sequences(
    frame: pd.DataFrame,
    *,
    context_length: int = 60,
    prediction_length: int = 5,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    feature_columns: Sequence[str] = DEFAULT_FEATURES,
    target_column: str = "Close",
) -> PreparedSequences:
    if context_length < 1 or prediction_length < 1:
        raise ValueError("Context and prediction lengths must be positive")
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if train_fraction + val_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than one")

    features = tuple(feature_columns)
    if not features or target_column not in features:
        raise ValueError("Target column must be included in feature columns")
    numeric = _validate_input(frame, features)
    row_count = len(numeric)
    train_end = int(row_count * train_fraction)
    val_end = int(row_count * (train_fraction + val_fraction))
    if train_end < context_length + prediction_length:
        raise ValueError("Training split has insufficient context and prediction rows")

    scaler = StandardScaler().fit(numeric.iloc[:train_end].to_numpy(dtype=np.float64))
    scaled = scaler.transform(numeric.to_numpy(dtype=np.float64)).astype(np.float32)
    sample_count = row_count - context_length - prediction_length + 1
    if sample_count < 1:
        raise ValueError("Input has insufficient context and prediction rows")

    inputs = np.lib.stride_tricks.sliding_window_view(
        scaled, context_length, axis=0
    )[:sample_count].swapaxes(1, 2)
    close = numeric[target_column].to_numpy(dtype=np.float64)
    starts = np.arange(sample_count)
    target_starts = starts + context_length
    target_rows = target_starts[:, None] + np.arange(prediction_length)
    anchors = close[target_starts - 1, None]
    targets = (close[target_rows] / anchors - 1.0).astype(np.float32)

    train_mask = target_rows[:, -1] < train_end
    val_mask = (target_starts >= train_end) & (target_rows[:, -1] < val_end)
    test_mask = target_starts >= val_end
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Each chronological split must contain at least one complete target")

    return PreparedSequences(
        train=SequenceDataset(inputs[train_mask], targets[train_mask]),
        val=SequenceDataset(inputs[val_mask], targets[val_mask]),
        test=SequenceDataset(inputs[test_mask], targets[test_mask]),
        scaler=scaler,
        feature_columns=features,
        train_end=train_end,
        val_end=val_end,
    )
