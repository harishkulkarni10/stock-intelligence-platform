"""LangGraph assembly and analyze_stock orchestrator."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

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


def build_graph():
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
    }


def _analyze_dto(
    *,
    ticker: str,
    forecast: dict[str, Any],
    graph_result: dict[str, Any],
    cached: bool = False,
) -> AnalyzeResult:
    return {
        "status": "ok",
        "ticker": ticker,
        "final_report": graph_result.get("final_report", ""),
        "recommendation": graph_result.get("recommendation", "NEUTRAL"),
        "confidence": graph_result.get("confidence", "Medium"),
        "performance_analysis": graph_result.get("performance_analysis", ""),
        "news_summary": graph_result.get("news_summary", ""),
        "predictions": _predictions_dto(forecast),
        "cached": cached,
    }


def analyze_stock(ticker: str, thread_id: str | None = None) -> AnalyzeResult:
    """Run forecast tools + agent graph; optionally reuse a Redis report cache hit."""
    del thread_id  # reserved for future multi-turn memory
    symbol = normalize_ticker(ticker)
    cache = ReportCache()
    cached = cache.get(symbol)
    if cached is not None:
        payload = dict(cached)
        payload["cached"] = True
        return payload  # type: ignore[return-value]

    forecast = get_forecast(symbol)
    if forecast.get("status") != "ok":
        return {
            "status": forecast.get("status", "error"),
            "ticker": symbol,
            "detail": forecast.get("error", "Forecast unavailable"),
            "predictions": {},
            "cached": False,
        }

    forecast_text = format_forecast_for_prompt(forecast)
    graph = build_graph()
    result = graph.invoke(
        {
            "ticker": symbol,
            "messages": [HumanMessage(content=f"Analyze {symbol}")],
            "forecast": forecast,
            "forecast_text": forecast_text,
        }
    )
    dto = _analyze_dto(ticker=symbol, forecast=forecast, graph_result=result, cached=False)
    cache.set(symbol, dto)
    return dto
