"""Smoke tests for company profile + history helpers."""

from __future__ import annotations

import pandas as pd

from src.market.equity import frame_to_history, get_company_profile


def test_frame_to_history_maps_ohlc():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "Open": [10.0, 11.0],
            "High": [12.0, 12.5],
            "Low": [9.5, 10.5],
            "Close": [11.0, 12.0],
            "Volume": [100, 110],
            "RSI14": [50.0, 51.0],
            "MACD": [0.1, 0.2],
        }
    )
    points = frame_to_history(frame)
    assert len(points) == 2
    assert points[0]["date"] == "2024-01-02"
    assert points[0]["close"] == 11.0
    assert points[1]["high"] == 12.5


def test_company_profile_handles_ticker_info(monkeypatch):
    class FakeTicker:
        info = {
            "shortName": "Example Corp",
            "sector": "Technology",
            "industry": "Software",
            "longBusinessSummary": "Example Corp builds analytics software for enterprises. "
            * 20,
            "exchange": "NMS",
            "website": "https://example.com",
        }

    monkeypatch.setattr("src.market.equity.yf.Ticker", lambda ticker: FakeTicker())
    profile = get_company_profile("exmp")
    assert profile["ticker"] == "EXMP"
    assert profile["name"] == "Example Corp"
    assert profile["sector"] == "Technology"
    assert profile["summary"]
    assert profile["summary"].endswith("…") or len(profile["summary"]) <= 900
