"""LangGraph assembly and analyze_stock orchestrator.

Production default: Performance Analyst only (agent 1) + LSTM/persistence forecast.
Full multi-agent graph remains available via build_full_graph() for later stages.
"""

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

# Bump when analyze DTO / agent set changes so stale Redis payloads are ignored.
ANALYZE_CACHE_PREFIX = "analyze-perf-v1"


def build_performance_graph():
    """Agent 1 only: forecast interpretation with guardrails."""
    graph = StateGraph(AgentState)
    graph.add_node("perf", performance_analyst_node)
    graph.set_entry_point("perf")
    graph.add_edge("perf", END)
    return graph.compile()


def build_full_graph():
    """Full pipeline (performance → news → report → critic). Not used by UI yet."""
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
    """Default production graph = agent 1 only."""
    return build_performance_graph()


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
    if source == "child":
        return "Medium"
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
    return {
        "status": "ok",
        "ticker": ticker,
        "mode": "performance_only",
        "final_report": analysis,
        "recommendation": _stance_from_trend(trend),
        "confidence": _confidence_for_forecast(forecast, repaired=repaired),
        "performance_analysis": analysis,
        "performance_trend": trend,
        "performance_guardrail_ok": graph_result.get("performance_guardrail_ok"),
        "performance_repaired": repaired,
        "news_summary": None,
        "news": {},
        "draft_report": None,
        "predictions": _predictions_dto(forecast),
        "cached": cached,
    }


def analyze_stock(ticker: str, thread_id: str | None = None) -> AnalyzeResult:
    """Run LSTM/persistence forecast + Performance Analyst only."""
    del thread_id  # reserved for future multi-turn memory
    symbol = normalize_ticker(ticker)
    cache = ReportCache(prefix=ANALYZE_CACHE_PREFIX)
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
            "mode": "performance_only",
            "detail": forecast.get("error", "Forecast unavailable"),
            "predictions": {},
            "cached": False,
        }

    forecast_text = format_forecast_for_prompt(forecast)
    graph = build_performance_graph()
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
