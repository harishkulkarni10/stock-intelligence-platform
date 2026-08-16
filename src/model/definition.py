"""Neural-network definitions used by the forecasting pipeline."""

from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Direct multi-horizon forecaster over a sequence of input features."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        layers: int = 2,
        horizon: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(input_size, hidden_size, layers, horizon) < 1:
            raise ValueError("model dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "layers": layers,
            "horizon": horizon,
            "dropout": dropout,
        }
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, lookback, features)")
        sequence, _ = self.lstm(inputs)
        return self.output(self.dropout(sequence[:, -1]))


# Short compatibility name for callers that use the architecture name directly.
LSTM = LSTMForecaster
