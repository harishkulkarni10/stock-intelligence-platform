"""LangGraph agent state and helpers."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import MessagesState


class AgentState(MessagesState, total=False):
    ticker: str
    forecast: dict[str, Any]
    forecast_text: str
    news: dict[str, Any]
    news_raw: str
    performance_analysis: str
    news_summary: str
    final_report: str
    recommendation: str
    confidence: str


class AnalyzeResult(TypedDict, total=False):
    status: str
    ticker: str
    final_report: str
    recommendation: str
    confidence: str
    performance_analysis: str
    news_summary: str
    predictions: dict[str, Any]
    cached: bool
    detail: str


def extract_stance_and_confidence(text: str) -> tuple[str, str]:
    upper = (text or "").upper()
    if "BULLISH" in upper:
        recommendation = "BULLISH"
    elif "BEARISH" in upper:
        recommendation = "BEARISH"
    else:
        recommendation = "NEUTRAL"

    if "CONFIDENCE: HIGH" in upper or "CONFIDENCE:** HIGH" in upper or "| **CONFIDENCE:** HIGH" in upper:
        confidence = "High"
    elif "CONFIDENCE: LOW" in upper or "CONFIDENCE:** LOW" in upper:
        confidence = "Low"
    elif "HIGH" in upper and "CONFIDENCE" in upper:
        confidence = "High"
    elif "LOW" in upper and "CONFIDENCE" in upper:
        confidence = "Low"
    else:
        confidence = "Medium"
    return recommendation, confidence
