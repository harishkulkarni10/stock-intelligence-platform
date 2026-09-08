"""Agent nodes: performance → news → report → critic."""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage

from logger.logger import get_logger
from src.agents.llm import get_chat_llm, message_text
from src.agents.news_guardrails import run_news_harness
from src.agents.performance_guardrails import run_performance_harness
from src.agents.state import extract_stance_and_confidence
from src.agents.tools import format_news_for_prompt, get_news

logger = get_logger()
llm = get_chat_llm()


def performance_analyst_node(state: dict) -> dict:
    ticker = state["ticker"]
    forecast = state.get("forecast") if isinstance(state.get("forecast"), dict) else None
    source = (forecast or {}).get("model_source", "unknown")
    n_points = len((forecast or {}).get("predictions") or [])
    logger.info(
        "[agent1] start ticker=%s model_source=%s forecast_points=%s",
        ticker,
        source,
        n_points,
    )

    harness = run_performance_harness(
        ticker=ticker,
        forecast_text=state.get("forecast_text", ""),
        forecast=forecast,
        llm=llm,
    )
    content = harness["performance_analysis"]
    logger.info(
        "[agent1] done ticker=%s trend=%s guardrail_ok=%s repaired=%s attempts=%s",
        ticker,
        harness["performance_trend"],
        harness["performance_guardrail_ok"],
        harness["performance_repaired"],
        len(harness.get("performance_attempts") or []),
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
    logger.info("[agent2] start ticker=%s", ticker)

    news = get_news(ticker)
    articles = news.get("articles") or []
    logger.info(
        "[agent2] news_fetch ticker=%s status=%s provider=%s articles=%s",
        ticker,
        news.get("status"),
        news.get("provider"),
        len(articles),
    )

    news_raw = format_news_for_prompt(news)
    harness = run_news_harness(
        ticker=ticker,
        news=news,
        news_raw=news_raw,
        llm=llm,
    )
    content = harness["news_summary"]
    logger.info(
        "[agent2] done ticker=%s sentiment=%s guardrail_ok=%s repaired=%s attempts=%s",
        ticker,
        harness["news_sentiment"],
        harness["news_guardrail_ok"],
        harness["news_repaired"],
        len(harness.get("news_attempts") or []),
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


def report_generator_node(state: dict) -> dict:
    ticker = state["ticker"]
    logger.info("[agent3] start ticker=%s", ticker)

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
    logger.info(
        "[agent3] done ticker=%s recommendation=%s confidence=%s",
        ticker,
        recommendation,
        confidence,
    )
    return {
        "messages": [AIMessage(content=text)],
        "draft_report": text,
        "recommendation": recommendation,
        "confidence": confidence,
    }


def critic_node(state: dict) -> dict:
    ticker = state.get("ticker", "")
    logger.info("[agent4] start ticker=%s", ticker)

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
    logger.info(
        "[agent4] done ticker=%s recommendation=%s confidence=%s",
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
