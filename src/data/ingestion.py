from __future__ import annotations

import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf

PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
OHLCV_COLUMNS = [*PRICE_COLUMNS, "Volume"]
FEATURE_COLUMNS = [*OHLCV_COLUMNS, "RSI14", "MACD"]
DEFAULT_FEATURE_PATH = (
    Path(__file__).resolve().parents[2] / "feature_store" / "data" / "features.parquet"
)
_WRITE_LOCK = threading.Lock()


def normalize_ticker(ticker: str) -> str:
    normalized = ticker.strip().upper().replace(".", "-")
    if not normalized or not re.fullmatch(r"[A-Z0-9^=-]+", normalized):
        raise ValueError(f"Invalid ticker: {ticker!r}")
    return normalized


def _flatten_download(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        for level in range(data.columns.nlevels):
            values = data.columns.get_level_values(level)
            if set(OHLCV_COLUMNS).issubset(values):
                data.columns = values
                break
        else:
            raise ValueError("Downloaded data does not contain OHLCV columns")

    missing = set(OHLCV_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {sorted(missing)}")

    data = data.loc[:, OHLCV_COLUMNS].copy()
    index_name = data.index.name or "Date"
    data = data.reset_index().rename(columns={index_name: "date", "Date": "date"})
    if "date" not in data:
        raise ValueError(f"Downloaded data for {ticker} has no date index")
    data["date"] = pd.to_datetime(data["date"], utc=True).dt.tz_localize(None)
    return data


def add_technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close = pd.to_numeric(data["Close"], errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    relative_strength = gain.div(loss)
    data["RSI14"] = 100 - (100 / (1 + relative_strength))
    data.loc[(loss == 0) & (gain > 0), "RSI14"] = 100.0
    data.loc[(loss == 0) & (gain == 0), "RSI14"] = 50.0
    data["MACD"] = close.ewm(span=12, adjust=False).mean() - close.ewm(
        span=26, adjust=False
    ).mean()
    return data.dropna(subset=["RSI14", "MACD"]).reset_index(drop=True)


def validate_features(
    frame: pd.DataFrame,
    *,
    context_length: int = 60,
    prediction_length: int = 5,
) -> None:
    required = ["date", *FEATURE_COLUMNS]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("OHLCV data is empty")
    if len(frame) < context_length + prediction_length:
        raise ValueError(
            f"Need at least {context_length + prediction_length} feature rows, got {len(frame)}"
        )

    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("Dates must be valid, strictly increasing, and unique")

    numeric = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("OHLCV and feature values must be numeric and finite")
    if (numeric[PRICE_COLUMNS] <= 0).any().any():
        raise ValueError("Prices must be positive")
    if (numeric["Volume"] < 0).any():
        raise ValueError("Volume must be nonnegative")
    if (numeric["High"] < numeric[PRICE_COLUMNS].max(axis=1)).any():
        raise ValueError("High must be at least Open, Low, and Close")
    if (numeric["Low"] > numeric[PRICE_COLUMNS].min(axis=1)).any():
        raise ValueError("Low must be at most Open, High, and Close")


def fetch_ohlcv(
    ticker: str,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    period: str | None = "5y",
    interval: str = "1d",
    context_length: int = 60,
    prediction_length: int = 5,
    downloader: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    symbol = normalize_ticker(ticker)
    download = downloader or yf.download
    kwargs: dict[str, object] = {
        "tickers": symbol,
        "interval": interval,
        "auto_adjust": False,
        "progress": False,
        "threads": False,
    }
    if start is not None or end is not None:
        kwargs.update(start=start, end=end)
    elif period is not None:
        kwargs["period"] = period

    raw = download(**kwargs)
    if raw is None or raw.empty:
        raise ValueError(f"No OHLCV data returned for {symbol}")
    data = add_technical_features(_flatten_download(raw, symbol))
    validate_features(
        data, context_length=context_length, prediction_length=prediction_length
    )
    return data.loc[:, ["date", *FEATURE_COLUMNS]]


def persist_features(
    frame: pd.DataFrame,
    ticker: str,
    path: str | Path = DEFAULT_FEATURE_PATH,
) -> Path:
    symbol = normalize_ticker(ticker)
    validate_features(frame, context_length=0, prediction_length=1)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    incoming = frame.loc[:, ["date", *FEATURE_COLUMNS]].copy()
    incoming = incoming.rename(columns={"date": "event_timestamp"})
    incoming[FEATURE_COLUMNS] = incoming[FEATURE_COLUMNS].astype("float32")
    incoming["ticker"] = symbol
    incoming["event_timestamp"] = pd.to_datetime(incoming["event_timestamp"], utc=True)
    incoming["created_timestamp"] = pd.Timestamp.now(tz="UTC")
    incoming = incoming.loc[
        :, ["ticker", "event_timestamp", "created_timestamp", *FEATURE_COLUMNS]
    ]

    with _WRITE_LOCK:
        if destination.exists():
            existing = pd.read_parquet(destination)
            combined = pd.concat([existing, incoming], ignore_index=True)
        else:
            combined = incoming
        combined["event_timestamp"] = pd.to_datetime(combined["event_timestamp"], utc=True)
        combined["created_timestamp"] = pd.to_datetime(combined["created_timestamp"], utc=True)
        combined = (
            combined.sort_values(["ticker", "event_timestamp", "created_timestamp"])
            .drop_duplicates(["ticker", "event_timestamp"], keep="last")
            .sort_values(["ticker", "event_timestamp"])
            .reset_index(drop=True)
        )

        handle, temporary_name = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
        )
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            combined.to_parquet(temporary, index=False)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def ingest_ticker(
    ticker: str,
    *,
    path: str | Path = DEFAULT_FEATURE_PATH,
    **fetch_kwargs: object,
) -> pd.DataFrame:
    frame = fetch_ohlcv(ticker, **fetch_kwargs)
    persist_features(frame, ticker, path)
    return frame
