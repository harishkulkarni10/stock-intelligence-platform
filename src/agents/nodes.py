"""Agent nodes: performance → news → report → critic."""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage

from logger.logger import get_logger
from src.agents.llm import get_chat_llm, message_text
from src.agents.performance_guardrails import run_performance_harness
from src.agents.state import extract_stance_and_confidence
from src.agents.tools import format_news_for_prompt, get_news

logger = get_logger()
llm = get_chat_llm()


def performance_analyst_node(state: dict) -> dict:
    ticker = state["ticker"]
    forecast_text = state.get("forecast_text", "")
    forecast = state.get("forecast")
    logger.info("performance_analyst ticker=%s", ticker)

    harness = run_performance_harness(
        ticker=ticker,
        forecast_text=forecast_text,
        forecast=forecast if isinstance(forecast, dict) else None,
        llm=llm,
    )
    content = harness["performance_analysis"]
    logger.info(
        "performance_analyst ticker=%s trend=%s guardrail_ok=%s repaired=%s",
        ticker,
        harness["performance_trend"],
        harness["performance_guardrail_ok"],
        harness["performance_repaired"],
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
    logger.info("market_expert ticker=%s", ticker)
    news = get_news(ticker)
    news_raw = format_news_for_prompt(news)

    prompt = f"""You are a market strategist summarizing news sentiment for {ticker}.

NEWS:
{news_raw}

Return a 3-5 line sentiment summary.
If news is unavailable or off-topic, say so explicitly.
Do not invent headlines.
"""
    response = llm.invoke([SystemMessage(content=prompt)])
    content = message_text(response)
    return {
        "messages": [AIMessage(content=content)],
        "news": news,
        "news_raw": news_raw,
        "news_summary": content,
    }


def report_generator_node(state: dict) -> dict:
    ticker = state["ticker"]
    logger.info("report_generator ticker=%s", ticker)

    prompt = f"""Write a clean Bloomberg-style markdown equity research note for {ticker}.

PERFORMANCE ANALYSIS:
{state.get("performance_analysis", "")}

NEWS SUMMARY:
{state.get("news_summary", "")}

FORECAST DATA:
{state.get("forecast_text", "")}

Rules:
- Ground price claims in FORECAST DATA only.
- Use PERFORMANCE ANALYSIS and NEWS SUMMARY explicitly.
- Keep it under 400 words.

End exactly with this line:
**Market Stance:** BULLISH/BEARISH/NEUTRAL | **Confidence:** High/Medium/Low
"""
    response = llm.invoke([SystemMessage(content=prompt)])
    text = message_text(response)
    recommendation, confidence = extract_stance_and_confidence(text)
    return {
        "messages": [AIMessage(content=text)],
        "draft_report": text,
        "recommendation": recommendation,
        "confidence": confidence,
    }


def critic_node(state: dict) -> dict:
    ticker = state.get("ticker", "")
    logger.info("critic ticker=%s", ticker)

    prompt = f"""You are a Senior Editor reviewing an equity research draft for {ticker}.

FORECAST DATA:
{state.get("forecast_text", "")}

PERFORMANCE ANALYSIS:
{state.get("performance_analysis", "")}

NEWS SUMMARY:
{state.get("news_summary", "")}

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
    return {
        "messages": [AIMessage(content=text)],
        "final_report": text,
        "recommendation": recommendation,
        "confidence": confidence,
    }
