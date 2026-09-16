"""LangGraph assembly and analyze_stock orchestrator.

Production default: Performance → Market Expert → Financial Analyst.
Report/critic remain available via build_full_graph().
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from logger.context import set_ticker
from logger.logger import get_logger, log_event
from src.agents.nodes import (
    critic_node,
    financial_analyst_node,
    market_expert_node,
    performance_analyst_node,
    report_generator_node,
)
from src.agents.llm import get_chat_llm
from src.agents.performance_guardrails import explain_summary_chips
from src.agents.state import AgentState, AnalyzeResult
from src.agents.tools import format_forecast_for_prompt, get_forecast
from src.data.ingestion import normalize_ticker
from src.market.equity import get_company_profile
from src.memory.semantic_cache import ReportCache

logger = get_logger("sip.analyze")

ANALYZE_CACHE_PREFIX = "analyze-perf-news-fin-v2"
ANALYZE_MODE = "performance_news_financial"


def build_performance_news_financial_graph():
    graph = StateGraph(AgentState)
    graph.add_node("perf", performance_analyst_node)
    graph.add_node("news", market_expert_node)
    graph.add_node("financial", financial_analyst_node)
    graph.set_entry_point("perf")
    graph.add_edge("perf", "news")
    graph.add_edge("news", "financial")
    graph.add_edge("financial", END)
    return graph.compile()


def build_performance_news_graph():
    """Legacy two-agent path kept for tests / rollback."""
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
    graph.add_node("financial", financial_analyst_node)
    graph.add_node("report", report_generator_node)
    graph.add_node("critic", critic_node)
    graph.set_entry_point("perf")
    graph.add_edge("perf", "news")
    graph.add_edge("news", "financial")
    graph.add_edge("financial", "report")
    graph.add_edge("report", "critic")
    graph.add_edge("critic", END)
    return graph.compile()


def build_graph():
    return build_performance_news_financial_graph()


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


def _financials_dto(payload: dict[str, Any] | None) -> dict[str, Any]:
    data = payload or {}
    return {
        "status": data.get("status"),
        "ticker": data.get("ticker"),
        "name": data.get("name"),
        "sector": data.get("sector"),
        "industry": data.get("industry"),
        "currency": data.get("currency"),
        "coverage": data.get("coverage"),
        "source": data.get("source"),
        "metrics": data.get("metrics") or {},
        "signals": data.get("signals") or {},
        "allowed_health": data.get("allowed_health"),
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
    company: dict[str, Any] | None = None,
    cached: bool = False,
    metric_explanations: dict[str, str] | None = None,
) -> AnalyzeResult:
    trend = graph_result.get("performance_trend")
    repaired = bool(graph_result.get("performance_repaired"))
    analysis = graph_result.get("performance_analysis", "")
    news_summary = graph_result.get("news_summary")
    confidence = _confidence_for_forecast(forecast, repaired=repaired)
    return {
        "status": "ok",
        "ticker": ticker,
        "mode": ANALYZE_MODE,
        "final_report": analysis,
        "recommendation": _stance_from_trend(trend),
        "confidence": confidence,
        "performance_analysis": analysis,
        "performance_trend": trend,
        "performance_guardrail_ok": graph_result.get("performance_guardrail_ok"),
        "performance_repaired": repaired,
        "news_summary": news_summary,
        "news_sentiment": graph_result.get("news_sentiment"),
        "news_guardrail_ok": graph_result.get("news_guardrail_ok"),
        "news_repaired": graph_result.get("news_repaired"),
        "news": graph_result.get("news") or {},
        "financial_analysis": graph_result.get("financial_analysis"),
        "financial_health": graph_result.get("financial_health"),
        "financial_guardrail_ok": graph_result.get("financial_guardrail_ok"),
        "financial_repaired": graph_result.get("financial_repaired"),
        "financials": _financials_dto(graph_result.get("financials")),
        "company": company or {},
        "draft_report": None,
        "predictions": _predictions_dto(forecast),
        "metric_explanations": metric_explanations or {},
        "cached": cached,
        "cached_at_ts": None,
        "cache_age_seconds": None,
        "cache_ttl_seconds": None,
    }


def analyze_stock(
    ticker: str,
    thread_id: str | None = None,
    *,
    force_refresh: bool = False,
) -> AnalyzeResult:
    """Champion forecast → A1 performance → A2 news → A3 financial."""
    del thread_id
    symbol = normalize_ticker(ticker)
    set_ticker(symbol)
    started = time.perf_counter()
    log_event(
        logger,
        "analyze_start",
        step="analyze_01_start",
        status="started",
        data={"ticker": symbol, "force_refresh": force_refresh},
    )

    cache = ReportCache(prefix=ANALYZE_CACHE_PREFIX)
    if force_refresh:
        cache.delete(symbol)
        log_event(
            logger,
            "analyze_cache_bypass",
            step="analyze_02_cache",
            status="bypass",
            data={"prefix": ANALYZE_CACHE_PREFIX},
        )
    else:
        entry = cache.read(symbol)
        if entry is not None:
            log_event(
                logger,
                "analyze_cache_hit",
                step="analyze_02_cache",
                status="hit",
                duration_ms=(time.perf_counter() - started) * 1000,
                data={
                    "prefix": ANALYZE_CACHE_PREFIX,
                    "age_seconds": entry.get("age_seconds"),
                    "ttl_seconds": entry.get("ttl_seconds"),
                },
            )
            payload = dict(entry["result"])
            payload["cached"] = True
            payload["cached_at_ts"] = entry.get("cached_at_ts")
            payload["cache_age_seconds"] = entry.get("age_seconds")
            payload["cache_ttl_seconds"] = entry.get("ttl_seconds")
            return payload  # type: ignore[return-value]

    log_event(
        logger,
        "analyze_cache_miss",
        step="analyze_02_cache",
        status="miss" if not force_refresh else "forced",
    )

    forecast_started = time.perf_counter()
    log_event(logger, "forecast_fetch", step="analyze_03_forecast", status="started")
    forecast = get_forecast(symbol)
    forecast_ms = (time.perf_counter() - forecast_started) * 1000
    if forecast.get("status") != "ok":
        log_event(
            logger,
            "forecast_failed",
            step="analyze_03_forecast",
            status="error",
            duration_ms=forecast_ms,
            data={
                "forecast_status": forecast.get("status"),
                "detail": forecast.get("error"),
            },
        )
        return {
            "status": forecast.get("status", "error"),
            "ticker": symbol,
            "mode": ANALYZE_MODE,
            "detail": forecast.get("error", "Forecast unavailable"),
            "predictions": {},
            "cached": False,
        }

    log_event(
        logger,
        "forecast_ok",
        step="analyze_03_forecast",
        status="ok",
        duration_ms=forecast_ms,
        data={
            "model_source": forecast.get("model_source"),
            "horizon": forecast.get("horizon"),
        },
    )
    forecast_text = format_forecast_for_prompt(forecast)

    profile_started = time.perf_counter()
    company = get_company_profile(symbol)
    log_event(
        logger,
        "company_profile",
        step="analyze_03b_profile",
        status=str(company.get("status") or "ok"),
        duration_ms=(time.perf_counter() - profile_started) * 1000,
        data={"has_summary": bool(company.get("summary"))},
    )

    graph_started = time.perf_counter()
    log_event(
        logger,
        "graph_invoke",
        step="analyze_04_graph",
        status="started",
        data={"mode": ANALYZE_MODE},
    )
    result = build_performance_news_financial_graph().invoke(
        {
            "ticker": symbol,
            "messages": [HumanMessage(content=f"Analyze {symbol}")],
            "forecast": forecast,
            "forecast_text": forecast_text,
        }
    )
    graph_ms = (time.perf_counter() - graph_started) * 1000

    repaired = bool(result.get("performance_repaired"))
    confidence = _confidence_for_forecast(forecast, repaired=repaired)
    chip_started = time.perf_counter()
    metric_explanations = explain_summary_chips(
        ticker=symbol,
        trend=result.get("performance_trend"),
        news_sentiment=result.get("news_sentiment"),
        confidence=confidence,
        forecast=forecast,
        performance_analysis=result.get("performance_analysis"),
        news_summary=result.get("news_summary"),
        llm=get_chat_llm(),
    )
    log_event(
        logger,
        "chip_explanations",
        step="analyze_04b_chips",
        status="ok",
        duration_ms=(time.perf_counter() - chip_started) * 1000,
        data={"keys": list(metric_explanations.keys())},
    )

    dto = _analyze_dto(
        ticker=symbol,
        forecast=forecast,
        graph_result=result,
        company=company,
        cached=False,
        metric_explanations=metric_explanations,
    )
    cache.set(symbol, dto)
    total_ms = (time.perf_counter() - started) * 1000
    log_event(
        logger,
        "analyze_complete",
        step="analyze_05_complete",
        status="ok",
        duration_ms=total_ms,
        metrics={
            "forecast_ms": round(forecast_ms, 2),
            "graph_ms": round(graph_ms, 2),
            "cached": False,
        },
        data={
            "trend": dto.get("performance_trend"),
            "sentiment": dto.get("news_sentiment"),
            "health": dto.get("financial_health"),
            "perf_guardrail": dto.get("performance_guardrail_ok"),
            "news_guardrail": dto.get("news_guardrail_ok"),
            "fin_guardrail": dto.get("financial_guardrail_ok"),
        },
    )
    return dto
