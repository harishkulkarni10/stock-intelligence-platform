from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import (
    FEATURE_COLUMNS,
    add_technical_features,
    fetch_ohlcv,
    persist_features,
    validate_features,
)
from src.data.preparation import DEFAULT_FEATURES, prepare_sequences


def raw_prices(rows: int = 180) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="D", name="Date")
    close = 100 + np.arange(rows, dtype=float) * 0.2 + np.sin(np.arange(rows) / 4)
    return pd.DataFrame(
        {
            "Open": close - 0.3,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000 + np.arange(rows),
        },
        index=index,
    )


def feature_frame(rows: int = 180) -> pd.DataFrame:
    raw = raw_prices(rows).reset_index().rename(columns={"Date": "date"})
    return add_technical_features(raw)


def test_fetch_normalizes_ticker_and_computes_features() -> None:
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        frame = raw_prices()
        frame.columns = pd.MultiIndex.from_product([frame.columns, ["BRK-B"]])
        return frame

    result = fetch_ohlcv(" brk.b ", downloader=download)

    assert calls[0]["tickers"] == "BRK-B"
    assert list(result.columns) == ["date", *FEATURE_COLUMNS]
    assert result[["RSI14", "MACD"]].notna().all().all()
    assert len(result) == 166


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.iloc[0:0], "empty"),
        (
            lambda data: pd.concat([data.iloc[[0]], data], ignore_index=True),
            "strictly increasing",
        ),
        (lambda data: data.assign(Close=np.inf), "finite"),
        (lambda data: data.assign(Open=-1), "positive"),
        (lambda data: data.assign(Volume=-1), "nonnegative"),
        (lambda data: data.assign(High=data["Close"] - 1), "High"),
        (lambda data: data.iloc[:64], "at least 65"),
    ],
)
def test_strict_validation_failures(mutate, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_features(mutate(feature_frame()))


def test_prepare_sequences_has_no_scaler_leakage_and_direct_returns() -> None:
    data = feature_frame(220)
    train_end = int(len(data) * 0.7)
    data.loc[data.index >= train_end, list(DEFAULT_FEATURES)] *= 1_000

    prepared = prepare_sequences(data)
    expected_train = data.loc[: train_end - 1, list(DEFAULT_FEATURES)].to_numpy()

    np.testing.assert_allclose(prepared.scaler.mean_, expected_train.mean(axis=0))
    assert len(prepared.train) > 0 and len(prepared.val) > 0 and len(prepared.test) > 0
    _, first_target = prepared.train[0]
    close = data["Close"].to_numpy()
    np.testing.assert_allclose(
        first_target.numpy(), close[60:65] / close[59] - 1, rtol=1e-6
    )


def test_sequence_splits_are_chronological() -> None:
    data = feature_frame(240)
    prepared = prepare_sequences(data)
    close_index = prepared.feature_columns.index("Close")

    def first_and_last(dataset):
        first = prepared.scaler.inverse_transform(dataset.inputs[0].numpy())[:, close_index]
        last = prepared.scaler.inverse_transform(dataset.inputs[-1].numpy())[:, close_index]
        return first[0], last[-1]

    train_range = first_and_last(prepared.train)
    val_range = first_and_last(prepared.val)
    test_range = first_and_last(prepared.test)
    assert train_range[0] < train_range[1] < val_range[1] < test_range[1]
    assert prepared.train.inputs.shape[1:] == (60, len(DEFAULT_FEATURES))
    assert prepared.train.targets.shape[1:] == (5,)


def test_persistence_deduplicates_with_latest_and_sorts(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    initial = feature_frame(100)
    persist_features(initial, "aapl", path)

    refreshed = initial.iloc[-3:].copy()
    refreshed["Close"] += 0.25
    refreshed["High"] += 0.25
    persist_features(refreshed, "AAPL", path)

    stored = pd.read_parquet(path)
    assert len(stored) == len(initial)
    assert not stored.duplicated(["ticker", "event_timestamp"]).any()
    assert stored["event_timestamp"].is_monotonic_increasing
    assert stored.iloc[-1]["Close"] == pytest.approx(refreshed.iloc[-1]["Close"])
    assert set(["ticker", "event_timestamp", "created_timestamp"]).issubset(stored.columns)
