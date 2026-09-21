"""Agent nodes: performance → news → financial → risk → report (critic optional)."""

from __future__ import annotations

import time

from langchain_core.messages import AIMessage, SystemMessage

from logger.logger import get_logger, log_event
from src.agents.financial_guardrails import run_financial_harness
from src.agents.llm import get_chat_llm, message_text
from src.agents.news_guardrails import run_news_harness
from src.agents.performance_guardrails import run_performance_harness
from src.agents.report_guardrails import run_report_harness
from src.agents.risk_guardrails import run_risk_harness
from src.agents.state import extract_stance_and_confidence
from src.agents.tools import (
    format_financials_for_prompt,
    format_news_for_prompt,
    get_financials,
    get_news,
)

logger = get_logger("sip.agents")
llm = get_chat_llm()


def performance_analyst_node(state: dict) -> dict:
    ticker = state["ticker"]
    forecast = state.get("forecast") if isinstance(state.get("forecast"), dict) else None
    source = (forecast or {}).get("model_source", "unknown")
    n_points = len((forecast or {}).get("predictions") or [])
    started = time.perf_counter()
    log_event(
        logger,
        "agent1_start",
        step="agent1",
        status="started",
        data={"model_source": source, "forecast_points": n_points},
    )

    harness = run_performance_harness(
        ticker=ticker,
        forecast_text=state.get("forecast_text", ""),
        forecast=forecast,
        llm=llm,
    )
    content = harness["performance_analysis"]
    log_event(
        logger,
        "agent1_done",
        step="agent1",
        status="success",
        duration_ms=(time.perf_counter() - started) * 1000,
        metrics={
            "repaired": harness["performance_repaired"],
            "attempts": len(harness.get("performance_attempts") or []),
            "guardrail_ok": harness["performance_guardrail_ok"],
        },
        data={"trend": harness["performance_trend"]},
    )
    return {
        "messages": [AIMessage(content=content)],
        "performance_analysis": content,
        "performance_trend": harness["performance_trend"],
        "performance_guardrail_ok": harness["performance_guardrail_ok"],
        "performance_repaired": harness["performance_repaired"],
    }


def market_expert_node(state: dict) -> dict:
    ticker = state["ticker"]
    started = time.perf_counter()
    log_event(logger, "agent2_start", step="agent2", status="started")

    news_started = time.perf_counter()
    news = get_news(ticker)
    news_ms = (time.perf_counter() - news_started) * 1000
    articles = news.get("articles") or []
    log_event(
        logger,
        "agent2_news_fetch",
        step="news_fetch",
        status="ok" if news.get("status") == "ok" else "error",
        duration_ms=news_ms,
        metrics={"articles": len(articles)},
        data={"provider": news.get("provider"), "news_status": news.get("status")},
    )

    news_raw = format_news_for_prompt(news)
    harness = run_news_harness(
        ticker=ticker,
        news=news,
        news_raw=news_raw,
        llm=llm,
    )
    content = harness["news_summary"]
    log_event(
        logger,
        "agent2_done",
        step="agent2",
        status="success",
        duration_ms=(time.perf_counter() - started) * 1000,
        metrics={
            "repaired": harness["news_repaired"],
            "attempts": len(harness.get("news_attempts") or []),
            "guardrail_ok": harness["news_guardrail_ok"],
        },
        data={"sentiment": harness["news_sentiment"]},
    )
    return {
        "messages": [AIMessage(content=content)],
        "news": news,
        "news_raw": news_raw,
        "news_summary": content,
        "news_sentiment": harness["news_sentiment"],
        "news_guardrail_ok": harness["news_guardrail_ok"],
        "news_repaired": harness["news_repaired"],
    }


def financial_analyst_node(state: dict) -> dict:
    ticker = state["ticker"]
    started = time.perf_counter()
    log_event(logger, "agent3_start", step="agent3", status="started")

    fin_started = time.perf_counter()
    financials = state.get("financials")
    if not isinstance(financials, dict):
        financials = get_financials(ticker)
    fin_ms = (time.perf_counter() - fin_started) * 1000
    log_event(
        logger,
        "agent3_financials_fetch",
        step="financials_fetch",
        status=str(financials.get("status") or "error"),
        duration_ms=fin_ms,
        data={
            "coverage": financials.get("coverage"),
            "allowed_health": financials.get("allowed_health"),
        },
    )

    financials_raw = format_financials_for_prompt(financials)
    harness = run_financial_harness(
        ticker=ticker,
        financials=financials,
        financials_raw=financials_raw,
        llm=llm,
    )
    content = harness["financial_analysis"]
    log_event(
        logger,
        "agent3_done",
        step="agent3",
        status="success",
        duration_ms=(time.perf_counter() - started) * 1000,
        metrics={
            "repaired": harness["financial_repaired"],
            "attempts": len(harness.get("financial_attempts") or []),
            "guardrail_ok": harness["financial_guardrail_ok"],
        },
        data={"health": harness["financial_health"]},
    )
    return {
        "messages": [AIMessage(content=content)],
        "financials": financials,
        "financials_raw": financials_raw,
        "financial_analysis": content,
        "financial_health": harness["financial_health"],
        "financial_guardrail_ok": harness["financial_guardrail_ok"],
        "financial_repaired": harness["financial_repaired"],
    }


def risk_analyst_node(state: dict) -> dict:
    ticker = state["ticker"]
    started = time.perf_counter()
    forecast = state.get("forecast") if isinstance(state.get("forecast"), dict) else None
    log_event(logger, "agent4_start", step="agent4", status="started")

    harness = run_risk_harness(
        ticker=ticker,
        forecast=forecast,
        performance_trend=state.get("performance_trend"),
        performance_repaired=bool(state.get("performance_repaired")),
        performance_analysis=state.get("performance_analysis"),
        news_sentiment=state.get("news_sentiment"),
        news_summary=state.get("news_summary"),
        financial_health=state.get("financial_health"),
        financial_analysis=state.get("financial_analysis"),
        financials=state.get("financials") if isinstance(state.get("financials"), dict) else None,
        llm=llm,
    )
    content = harness["risk_analysis"]
    log_event(
        logger,
        "agent4_done",
        step="agent4",
        status="success",
        duration_ms=(time.perf_counter() - started) * 1000,
        metrics={
            "repaired": harness["risk_repaired"],
            "attempts": len(harness.get("risk_attempts") or []),
            "guardrail_ok": harness["risk_guardrail_ok"],
        },
        data={"risk": harness["risk_level"]},
    )
    return {
        "messages": [AIMessage(content=content)],
        "risk": harness.get("risk") or {},
        "risk_analysis": content,
        "risk_level": harness["risk_level"],
        "risk_guardrail_ok": harness["risk_guardrail_ok"],
        "risk_repaired": harness["risk_repaired"],
    }


def report_generator_node(state: dict) -> dict:
    ticker = state["ticker"]
    started = time.perf_counter()
    forecast = state.get("forecast") if isinstance(state.get("forecast"), dict) else None
    source = str((forecast or {}).get("model_source") or "").lower()
    repaired = bool(state.get("performance_repaired"))
    confidence = "Low" if repaired or source == "persistence" else "Medium"
    log_event(logger, "agent5_start", step="agent5", status="started")

    harness = run_report_harness(
        ticker=ticker,
        forecast=forecast,
        forecast_text=state.get("forecast_text") or "",
        performance_trend=state.get("performance_trend"),
        performance_analysis=state.get("performance_analysis") or "",
        news_sentiment=state.get("news_sentiment"),
        news_summary=state.get("news_summary") or "",
        financial_health=state.get("financial_health"),
        financial_analysis=state.get("financial_analysis") or "",
        financials=state.get("financials") if isinstance(state.get("financials"), dict) else None,
        risk_level=state.get("risk_level"),
        risk_analysis=state.get("risk_analysis") or "",
        risk=state.get("risk") if isinstance(state.get("risk"), dict) else None,
        confidence=confidence,
        llm=llm,
    )
    content = harness["final_report"]
    log_event(
        logger,
        "agent5_done",
        step="agent5",
        status="success",
        duration_ms=(time.perf_counter() - started) * 1000,
        metrics={
            "repaired": harness["report_repaired"],
            "attempts": len(harness.get("report_attempts") or []),
            "guardrail_ok": harness["report_guardrail_ok"],
        },
        data={
            "stance": harness["recommendation"],
            "confidence": harness["confidence"],
        },
    )
    return {
        "messages": [AIMessage(content=content)],
        "draft_report": harness["draft_report"],
        "final_report": content,
        "recommendation": harness["recommendation"],
        "confidence": harness["confidence"],
        "report_guardrail_ok": harness["report_guardrail_ok"],
        "report_repaired": harness["report_repaired"],
    }


def critic_node(state: dict) -> dict:
    ticker = state.get("ticker", "")
    logger.info("[critic] start ticker=%s", ticker)

    prompt = f"""You are a Senior Editor reviewing an equity research draft for {ticker}.

FORECAST DATA:
{state.get("forecast_text", "")}

PERFORMANCE ANALYSIS:
{state.get("performance_analysis", "")}

NEWS SUMMARY:
{state.get("news_summary", "")}

FINANCIAL ANALYSIS:
{state.get("financial_analysis", "")}

DRAFT REPORT:
{state.get("draft_report", "")}

Tasks:
1) Check Market Stance vs forecast direction.
2) Remove unsupported numeric claims.
3) Keep Bloomberg tone.
4) Output ONLY the final markdown report.

End exactly with:
**Market Stance:** BULLISH/BEARISH/NEUTRAL | **Confidence:** High/Medium/Low
"""
    response = llm.invoke([SystemMessage(content=prompt)])
    text = message_text(response)
    recommendation, confidence = extract_stance_and_confidence(text)
    logger.info(
        "[critic] done ticker=%s recommendation=%s confidence=%s",
        ticker,
        recommendation,
        confidence,
    )
    return {
        "messages": [AIMessage(content=text)],
        "final_report": text,
        "recommendation": recommendation,
        "confidence": confidence,
    }
