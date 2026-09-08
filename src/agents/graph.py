"""LangGraph assembly and analyze_stock orchestrator.

Production default: Performance Analyst → Market Expert on champion forecasts.
Report/critic remain available via build_full_graph().
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from logger.logger import get_logger
from src.agents.nodes import (
    critic_node,
    market_expert_node,
    performance_analyst_node,
    report_generator_node,
)
from src.agents.state import AgentState, AnalyzeResult
from src.agents.tools import format_forecast_for_prompt, get_forecast
from src.data.ingestion import normalize_ticker
from src.memory.semantic_cache import ReportCache

logger = get_logger()

ANALYZE_CACHE_PREFIX = "analyze-perf-news-v2"


def build_performance_news_graph():
    graph = StateGraph(AgentState)
    graph.add_node("perf", performance_analyst_node)
    graph.add_node("news", market_expert_node)
    graph.set_entry_point("perf")
    graph.add_edge("perf", "news")
    graph.add_edge("news", END)
    return graph.compile()


def build_performance_graph():
    graph = StateGraph(AgentState)
    graph.add_node("perf", performance_analyst_node)
    graph.set_entry_point("perf")
    graph.add_edge("perf", END)
    return graph.compile()


def build_full_graph():
    graph = StateGraph(AgentState)
    graph.add_node("perf", performance_analyst_node)
    graph.add_node("news", market_expert_node)
    graph.add_node("report", report_generator_node)
    graph.add_node("critic", critic_node)
    graph.set_entry_point("perf")
    graph.add_edge("perf", "news")
    graph.add_edge("news", "report")
    graph.add_edge("report", "critic")
    graph.add_edge("critic", END)
    return graph.compile()


def build_graph():
    return build_performance_news_graph()


def _predictions_dto(forecast: dict[str, Any]) -> dict[str, Any]:
    return {
        "forecast": forecast.get("predictions", []),
        "history": forecast.get("history", []),
        "last_close": forecast.get("last_close"),
        "last_date": forecast.get("last_date"),
        "model_version": forecast.get("model_version"),
        "model_type": forecast.get("model_type"),
        "model_source": forecast.get("model_source"),
        "horizon": forecast.get("horizon"),
        "evaluation": forecast.get("evaluation") or {},
        "champion": forecast.get("champion"),
        "beats_persistence": forecast.get("beats_persistence"),
    }


def _stance_from_trend(trend: str | None) -> str:
    label = (trend or "SIDEWAYS").upper()
    if label == "SIDEWAYS":
        return "NEUTRAL"
    if label in {"BULLISH", "BEARISH"}:
        return label
    return "NEUTRAL"


def _confidence_for_forecast(forecast: dict[str, Any], *, repaired: bool) -> str:
    source = str(forecast.get("model_source") or "").lower()
    if repaired or source == "persistence":
        return "Low"
    return "Medium"


def _analyze_dto(
    *,
    ticker: str,
    forecast: dict[str, Any],
    graph_result: dict[str, Any],
    cached: bool = False,
) -> AnalyzeResult:
    trend = graph_result.get("performance_trend")
    repaired = bool(graph_result.get("performance_repaired"))
    analysis = graph_result.get("performance_analysis", "")
    news_summary = graph_result.get("news_summary")
    return {
        "status": "ok",
        "ticker": ticker,
        "mode": "performance_news",
        "final_report": analysis,
        "recommendation": _stance_from_trend(trend),
        "confidence": _confidence_for_forecast(forecast, repaired=repaired),
        "performance_analysis": analysis,
        "performance_trend": trend,
        "performance_guardrail_ok": graph_result.get("performance_guardrail_ok"),
        "performance_repaired": repaired,
        "news_summary": news_summary,
        "news_sentiment": graph_result.get("news_sentiment"),
        "news_guardrail_ok": graph_result.get("news_guardrail_ok"),
        "news_repaired": graph_result.get("news_repaired"),
        "news": graph_result.get("news") or {},
        "draft_report": None,
        "predictions": _predictions_dto(forecast),
        "cached": cached,
    }


def analyze_stock(ticker: str, thread_id: str | None = None) -> AnalyzeResult:
    """Champion forecast → agent1 (performance) → agent2 (news)."""
    del thread_id
    symbol = normalize_ticker(ticker)
    logger.info("[analyze] step=01 start ticker=%s", symbol)

    cache = ReportCache(prefix=ANALYZE_CACHE_PREFIX)
    cached = cache.get(symbol)
    if cached is not None:
        logger.info("[analyze] step=02 cache_hit ticker=%s prefix=%s", symbol, ANALYZE_CACHE_PREFIX)
        payload = dict(cached)
        payload["cached"] = True
        return payload  # type: ignore[return-value]

    logger.info("[analyze] step=02 cache_miss ticker=%s", symbol)
    logger.info("[analyze] step=03 forecast_fetch ticker=%s", symbol)
    forecast = get_forecast(symbol)
    if forecast.get("status") != "ok":
        logger.warning(
            "[analyze] step=03 forecast_failed ticker=%s status=%s detail=%s",
            symbol,
            forecast.get("status"),
            forecast.get("error"),
        )
        return {
            "status": forecast.get("status", "error"),
            "ticker": symbol,
            "mode": "performance_news",
            "detail": forecast.get("error", "Forecast unavailable"),
            "predictions": {},
            "cached": False,
        }

    logger.info(
        "[analyze] step=03 forecast_ok ticker=%s model_source=%s horizon=%s",
        symbol,
        forecast.get("model_source"),
        forecast.get("horizon"),
    )
    forecast_text = format_forecast_for_prompt(forecast)
    logger.info("[analyze] step=04 graph_invoke mode=performance_news ticker=%s", symbol)
    result = build_performance_news_graph().invoke(
        {
            "ticker": symbol,
            "messages": [HumanMessage(content=f"Analyze {symbol}")],
            "forecast": forecast,
            "forecast_text": forecast_text,
        }
    )
    dto = _analyze_dto(ticker=symbol, forecast=forecast, graph_result=result, cached=False)
    cache.set(symbol, dto)
    logger.info(
        "[analyze] step=05 complete ticker=%s trend=%s sentiment=%s "
        "perf_guardrail=%s news_guardrail=%s cached=false",
        symbol,
        dto.get("performance_trend"),
        dto.get("news_sentiment"),
        dto.get("performance_guardrail_ok"),
        dto.get("news_guardrail_ok"),
    )
    return dto
