"""Small, dependency-light forecast decoding helpers."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import torch


def decode_forecast(
    predicted_cumulative_returns: np.ndarray | torch.Tensor | list[float],
    last_close: float,
    last_date: str | date | datetime | pd.Timestamp,
    *,
    transform: str = "simple",
) -> pd.Series:
    """Return forecast prices indexed by business days after ``last_date``."""
    if last_close <= 0:
        raise ValueError("last_close must be positive")
    if isinstance(predicted_cumulative_returns, torch.Tensor):
        predicted_cumulative_returns = (
            predicted_cumulative_returns.detach().cpu().numpy()
        )
    returns = np.asarray(predicted_cumulative_returns, dtype=float).squeeze()
    if returns.ndim != 1 or returns.size == 0:
        raise ValueError("predicted returns must be a non-empty one-dimensional horizon")
    if transform == "simple":
        prices = last_close * (1.0 + returns)
    elif transform == "log":
        prices = last_close * np.exp(returns)
    else:
        raise ValueError("transform must be 'simple' or 'log'")
    dates = pd.bdate_range(start=pd.Timestamp(last_date) + pd.offsets.BDay(1), periods=returns.size)
    return pd.Series(prices, index=dates, name="forecast_price")


forecast_prices = decode_forecast
