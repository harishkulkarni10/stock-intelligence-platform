"""Deterministic tools the agent graph calls for forecasts and news.

Numbers and headlines come from code/APIs — not from the LLM.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import requests
import yfinance as yf
from dotenv import load_dotenv

from logger.logger import get_logger
from src.data.ingestion import normalize_ticker
from src.pipelines.inference_pipeline import predict_best
from src.pipelines.training_pipeline import child_artifact_dir, parent_artifact_dir

load_dotenv()

FINNHUB_API_KEY = os.getenv("FMI_API_KEY") or os.getenv("FINNHUB_API_KEY")
FINNHUB_URL = "https://finnhub.io/api/v1/company-news"
_log = get_logger()

# Company aliases for ticker relevance (Yahoo feeds are often unrelated).
_TICKER_ALIASES: dict[str, tuple[str, ...]] = {
    "NVDA": ("nvidia", "geforce", "cuda"),
    "AAPL": ("apple", "iphone", "ipad", "macbook"),
    "MSFT": ("microsoft", "azure", "xbox", "openai"),
    "TSLA": ("tesla", "cybertruck", "elon musk"),
    "AMZN": ("amazon", "aws", "prime"),
    "GOOGL": ("alphabet", "google", "youtube"),
    "GOOG": ("alphabet", "google", "youtube"),
    "META": ("meta", "facebook", "instagram", "whatsapp"),
    "AMD": ("advanced micro devices", "radeon", "epyc"),
    "INTC": ("intel", "foundry"),
    "NFLX": ("netflix"),
    "AVGO": ("broadcom"),
    "ORCL": ("oracle"),
}


def _parent_exists() -> bool:
    return (parent_artifact_dir() / "model.pt").exists()


def _child_exists(ticker: str) -> bool:
    return (child_artifact_dir(ticker) / "model.pt").exists()


def get_forecast(ticker: str, horizon: int = 5) -> dict[str, Any]:
    """Load the evaluation winner and run live inference in-process.

    Prefers a promoted child artifact; otherwise serves the champion from the
    latest train summary (parent or persistence). Never starts training here.
    """
    symbol = normalize_ticker(ticker)
    if horizon < 1:
        raise ValueError("horizon must be positive")

    if not _parent_exists() and not _child_exists(symbol):
        return {
            "status": "missing_model",
            "ticker": symbol,
            "horizon": horizon,
            "error": "No parent or child model artifact found under outputs/",
        }

    payload = predict_best(symbol, horizon)
    source = payload.get("model_source", "parent")
    points = [
        {
            "step": int(item["step"]),
            "date": item.get("date"),
            "value": float(item.get("value", item.get("close"))),
        }
        for item in payload.get("predictions", [])
    ]
    evaluation = payload.get("evaluation") or {}
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
        "evaluation": evaluation,
        "champion": evaluation.get("champion", source),
        "beats_persistence": evaluation.get("beats_persistence"),
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


def _alias_terms(ticker: str) -> tuple[str, ...]:
    symbol = normalize_ticker(ticker)
    extras = _TICKER_ALIASES.get(symbol, ())
    return (symbol.lower(),) + tuple(term.lower() for term in extras)


def article_mentions_ticker(article: dict[str, Any], ticker: str) -> bool:
    """True when headline/summary clearly references the ticker or company aliases."""
    symbol = normalize_ticker(ticker)
    blob = f"{article.get('headline') or ''} {article.get('summary') or ''}".lower()
    if not blob.strip():
        return False
    if re.search(rf"\b{re.escape(symbol.lower())}\b", blob):
        return True
    for term in _alias_terms(symbol)[1:]:
        if term and term in blob:
            return True
    return False


def filter_ticker_relevant_articles(
    articles: list[dict[str, Any]],
    ticker: str,
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    """Keep only ticker-relevant articles; return (kept[:limit], dropped_count)."""
    relevant = [article for article in articles if article_mentions_ticker(article, ticker)]
    dropped = len(articles) - len(relevant)
    return relevant[:limit], dropped


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
    # Pull a wider window; relevance filter trims to `limit`.
    for article in articles[: max(limit * 4, 20)]:
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
    fetch_cap = max(limit * 5, 25)
    for entry in raw[:fetch_cap]:
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
    """Fetch recent headlines and keep only ticker-relevant ones.

    Finnhub company-news first when keyed; Yahoo otherwise. Irrelevant Yahoo
    market blurbs are dropped so agent 2 does not invent a ticker narrative.
    """
    symbol = normalize_ticker(ticker)
    errors: list[str] = []
    _log.info(
        "[news_tool] fetch_start ticker=%s limit=%s finnhub_keyed=%s",
        symbol,
        limit,
        bool(FINNHUB_API_KEY),
    )

    raw_articles: list[dict[str, Any]] = []
    provider: str | None = None

    if FINNHUB_API_KEY:
        try:
            raw_articles = _news_from_finnhub(symbol, limit=limit)
            provider = "finnhub"
        except Exception as exc:  # noqa: BLE001 - fall through to Yahoo
            errors.append(f"finnhub: {exc}")
            _log.warning("[news_tool] finnhub_failed ticker=%s error=%s", symbol, exc)

    if not raw_articles:
        try:
            raw_articles = _news_from_yahoo(symbol, limit=limit)
            provider = "yahoo"
        except Exception as exc:  # noqa: BLE001 - surface both failures
            errors.append(f"yahoo: {exc}")
            _log.error(
                "[news_tool] fetch_failed ticker=%s errors=%s",
                symbol,
                "; ".join(errors),
            )
            return {
                "status": "error",
                "ticker": symbol,
                "provider": None,
                "articles": [],
                "error": "; ".join(errors),
            }

    articles, dropped = filter_ticker_relevant_articles(
        raw_articles, symbol, limit=limit
    )
    _log.info(
        "[news_tool] relevance_filter ticker=%s provider=%s raw=%s kept=%s dropped=%s",
        symbol,
        provider,
        len(raw_articles),
        len(articles),
        dropped,
    )

    if not articles:
        _log.warning(
            "[news_tool] no_relevant_headlines ticker=%s provider=%s raw=%s",
            symbol,
            provider,
            len(raw_articles),
        )
        return {
            "status": "error",
            "ticker": symbol,
            "provider": provider,
            "articles": [],
            "raw_count": len(raw_articles),
            "filtered_out": len(raw_articles),
            "error": "no_ticker_relevant_headlines",
            "warnings": errors,
        }

    _log.info(
        "[news_tool] fetch_ok ticker=%s provider=%s articles=%s",
        symbol,
        provider,
        len(articles),
    )
    return {
        "status": "ok",
        "ticker": symbol,
        "provider": provider,
        "articles": articles,
        "raw_count": len(raw_articles),
        "filtered_out": max(0, len(raw_articles) - len(articles)),
        "warnings": errors,
    }


def format_news_for_prompt(news: dict[str, Any]) -> str:
    """Compact text block for LLM prompts."""
    if news.get("status") != "ok":
        detail = news.get("error") or "unavailable"
        return (
            f"News unavailable for {news.get('ticker')}: {detail}. "
            "Do not invent headlines."
        )

    lines = [
        f"Ticker-relevant news for {news['ticker']} ({news.get('provider')}):"
    ]
    for article in news.get("articles", []):
        lines.append(
            f"- ({article.get('date')}) {article.get('headline')}\n"
            f"  {article.get('summary')}\n"
            f"  {article.get('url')}"
        )
    return "\n".join(lines)
