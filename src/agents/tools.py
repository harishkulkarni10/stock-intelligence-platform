"""Deterministic tools the agent graph calls for forecasts and news.

Numbers and headlines come from code/APIs — not from the LLM.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

import requests
import yfinance as yf
from dotenv import load_dotenv

from src.data.ingestion import normalize_ticker
from src.pipelines.inference_pipeline import predict_child, predict_parent
from src.pipelines.training_pipeline import child_artifact_dir, parent_artifact_dir

load_dotenv()

FINNHUB_API_KEY = os.getenv("FMI_API_KEY") or os.getenv("FINNHUB_API_KEY")
FINNHUB_URL = "https://finnhub.io/api/v1/company-news"


def _parent_exists() -> bool:
    return (parent_artifact_dir() / "model.pt").exists()


def _child_exists(ticker: str) -> bool:
    return (child_artifact_dir(ticker) / "model.pt").exists()


def get_forecast(ticker: str, horizon: int = 5) -> dict[str, Any]:
    """Load the best available artifact and run live inference in-process.

    Prefers a promoted child model; otherwise uses the parent weights on the
    ticker's recent features. Never starts training from the agent path.
    """
    symbol = normalize_ticker(ticker)
    if horizon < 1:
        raise ValueError("horizon must be positive")

    if _child_exists(symbol):
        payload = predict_child(symbol, horizon)
        source = "child"
    elif _parent_exists():
        payload = predict_parent(symbol, horizon)
        source = "parent"
    else:
        return {
            "status": "missing_model",
            "ticker": symbol,
            "horizon": horizon,
            "error": "No parent or child model artifact found under outputs/",
        }

    points = [
        {
            "step": int(item["step"]),
            "date": item.get("date"),
            "value": float(item.get("value", item.get("close"))),
        }
        for item in payload.get("predictions", [])
    ]
    return {
        "status": "ok",
        "ticker": symbol,
        "horizon": int(payload.get("horizon", horizon)),
        "model_source": source,
        "model_version": payload.get("model_version"),
        "model_type": payload.get("model_type"),
        "last_close": payload.get("last_close"),
        "last_date": payload.get("last_date"),
        "history": payload.get("history", []),
        "predictions": points,
        "artifact_dir": payload.get("artifact_dir"),
    }


def format_forecast_for_prompt(forecast: dict[str, Any]) -> str:
    """Compact text block for LLM prompts."""
    if forecast.get("status") != "ok":
        return (
            f"Forecast unavailable for {forecast.get('ticker')}: "
            f"{forecast.get('error') or forecast.get('status')}"
        )

    lines = [
        f"Forecast for {forecast['ticker']} "
        f"(model={forecast.get('model_source')}, version={forecast.get('model_version')}):",
        f"Last close {forecast.get('last_close')} on {forecast.get('last_date')}.",
        "Predicted closes:",
    ]
    for point in forecast.get("predictions", []):
        lines.append(f"  step {point['step']} ({point.get('date')}): {point['value']:.4f}")
    return "\n".join(lines)


def _news_from_finnhub(ticker: str, *, limit: int = 5) -> list[dict[str, str]]:
    if not FINNHUB_API_KEY:
        raise RuntimeError("Finnhub API key not set")

    end = datetime.now(UTC).date()
    start = end - timedelta(days=7)
    response = requests.get(
        FINNHUB_URL,
        params={
            "symbol": ticker,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "token": FINNHUB_API_KEY,
        },
        timeout=30,
    )
    response.raise_for_status()
    articles = response.json()
    if not isinstance(articles, list) or not articles:
        raise RuntimeError("Finnhub returned no articles")

    items: list[dict[str, str]] = []
    for article in articles[:limit]:
        stamp = article.get("datetime") or 0
        date = datetime.fromtimestamp(int(stamp), tz=UTC).strftime("%Y-%m-%d")
        items.append(
            {
                "source": "finnhub",
                "date": date,
                "headline": str(article.get("headline") or "").strip(),
                "summary": str(article.get("summary") or "").strip(),
                "url": str(article.get("url") or "").strip(),
            }
        )
    if not any(item["headline"] for item in items):
        raise RuntimeError("Finnhub articles missing headlines")
    return items


def _news_from_yahoo(ticker: str, *, limit: int = 5) -> list[dict[str, str]]:
    raw = yf.Ticker(ticker).news or []
    items: list[dict[str, str]] = []
    for entry in raw[:limit]:
        content = entry.get("content") if isinstance(entry.get("content"), dict) else entry
        provider = content.get("provider") if isinstance(content, dict) else {}
        title = (
            (content or {}).get("title")
            or entry.get("title")
            or ""
        )
        summary = (content or {}).get("summary") or entry.get("summary") or ""
        url = ""
        click = (content or {}).get("clickThroughUrl") or {}
        if isinstance(click, dict):
            url = str(click.get("url") or "")
        url = url or str(entry.get("link") or entry.get("url") or "")
        published = (
            (content or {}).get("pubDate")
            or entry.get("providerPublishTime")
            or ""
        )
        if isinstance(published, (int, float)):
            published = datetime.fromtimestamp(int(published), tz=UTC).strftime("%Y-%m-%d")
        else:
            published = str(published)[:10]
        items.append(
            {
                "source": "yahoo",
                "date": published,
                "headline": str(title).strip(),
                "summary": str(summary).strip(),
                "url": url.strip(),
                "publisher": str((provider or {}).get("displayName") or ""),
            }
        )
    if not items:
        raise RuntimeError("Yahoo returned no news")
    return items


def get_news(ticker: str, *, limit: int = 5) -> dict[str, Any]:
    """Fetch recent headlines. Finnhub first when keyed; Yahoo otherwise."""
    symbol = normalize_ticker(ticker)
    errors: list[str] = []

    if FINNHUB_API_KEY:
        try:
            articles = _news_from_finnhub(symbol, limit=limit)
            return {
                "status": "ok",
                "ticker": symbol,
                "provider": "finnhub",
                "articles": articles,
            }
        except Exception as exc:  # noqa: BLE001 - fall through to Yahoo
            errors.append(f"finnhub: {exc}")

    try:
        articles = _news_from_yahoo(symbol, limit=limit)
        return {
            "status": "ok",
            "ticker": symbol,
            "provider": "yahoo",
            "articles": articles,
            "warnings": errors,
        }
    except Exception as exc:  # noqa: BLE001 - surface both failures
        errors.append(f"yahoo: {exc}")
        return {
            "status": "error",
            "ticker": symbol,
            "provider": None,
            "articles": [],
            "error": "; ".join(errors),
        }


def format_news_for_prompt(news: dict[str, Any]) -> str:
    """Compact text block for LLM prompts."""
    if news.get("status") != "ok":
        return f"News unavailable for {news.get('ticker')}: {news.get('error')}"

    lines = [f"Latest news for {news['ticker']} ({news.get('provider')}):"]
    for article in news.get("articles", []):
        lines.append(
            f"- ({article.get('date')}) {article.get('headline')}\n"
            f"  {article.get('summary')}\n"
            f"  {article.get('url')}"
        )
    return "\n".join(lines)
