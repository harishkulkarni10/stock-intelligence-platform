"""Market data helpers: company profile and chart history (Yahoo / yfinance)."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import yfinance as yf

from logger.logger import get_logger, log_event
from src.data.ingestion import normalize_ticker

logger = get_logger("sip.market")

_SUMMARY_MAX_CHARS = 900


def _trim_summary(text: str, *, max_chars: int = _SUMMARY_MAX_CHARS) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[: max_chars].rsplit(" ", 1)[0].rstrip(" ,;")
    return f"{cut}…"


def get_company_profile(ticker: str) -> dict[str, Any]:
    """Fetch a short company overview (no founder/CEO focus)."""
    symbol = normalize_ticker(ticker)
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as exc:  # noqa: BLE001 - profile is optional UI enrichment
        log_event(
            logger,
            "company_profile_failed",
            step="company_profile",
            status="error",
            data={"ticker": symbol, "error": str(exc)},
        )
        return {
            "ticker": symbol,
            "name": symbol,
            "sector": None,
            "industry": None,
            "summary": None,
            "exchange": None,
            "website": None,
            "status": "error",
            "error": str(exc),
        }

    name = (
        str(info.get("shortName") or info.get("longName") or symbol).strip() or symbol
    )
    summary_raw = str(info.get("longBusinessSummary") or "").strip()
    summary = _trim_summary(summary_raw) if summary_raw else None
    payload = {
        "ticker": symbol,
        "name": name,
        "sector": (str(info.get("sector")).strip() if info.get("sector") else None),
        "industry": (
            str(info.get("industry")).strip() if info.get("industry") else None
        ),
        "summary": summary,
        "exchange": (
            str(info.get("exchange") or info.get("fullExchangeName") or "").strip()
            or None
        ),
        "website": (str(info.get("website")).strip() if info.get("website") else None),
        "status": "ok" if summary else "partial",
    }
    log_event(
        logger,
        "company_profile_ok",
        step="company_profile",
        status=payload["status"],
        data={
            "ticker": symbol,
            "has_summary": bool(summary),
            "sector": payload["sector"],
        },
    )
    return payload


def frame_to_history(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Convert an OHLCV feature frame into chart-ready history points."""
    if frame is None or frame.empty:
        return []
    points: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        date = getattr(row, "date", None)
        close = getattr(row, "Close", None)
        if date is None or close is None:
            continue
        try:
            point = {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "close": float(close),
                "open": float(getattr(row, "Open", close)),
                "high": float(getattr(row, "High", close)),
                "low": float(getattr(row, "Low", close)),
            }
        except (TypeError, ValueError):
            continue
        points.append(point)
    return points
