from __future__ import annotations

from src.agents.state import extract_stance_and_confidence


def test_extract_stance_and_confidence():
    text = "Outlook is constructive.\n**Market Stance:** BULLISH | **Confidence:** High"
    assert extract_stance_and_confidence(text) == ("BULLISH", "High")


def test_analyze_stock_missing_model(monkeypatch):
    from src.agents import graph

    monkeypatch.setattr(
        graph,
        "get_forecast",
        lambda ticker: {"status": "missing_model", "ticker": ticker, "error": "none"},
    )
    monkeypatch.setattr(graph.ReportCache, "get", lambda self, ticker: None)

    result = graph.analyze_stock("NVDA")

    assert result["status"] == "missing_model"
    assert result["ticker"] == "NVDA"


def test_analyze_stock_happy_path(monkeypatch):
    from src.agents import graph

    forecast = {
        "status": "ok",
        "ticker": "NVDA",
        "horizon": 5,
        "model_source": "parent",
        "model_version": "parent-1",
        "model_type": "parent",
        "last_close": 100.0,
        "last_date": "2026-08-14",
        "history": [],
        "predictions": [{"step": 1, "date": "2026-08-17", "value": 101.0}],
    }

    class FakeGraph:
        def invoke(self, state):
            return {
                "final_report": "Report\n**Market Stance:** BULLISH | **Confidence:** Medium",
                "recommendation": "BULLISH",
                "confidence": "Medium",
                "performance_analysis": "Upward bias",
                "news_summary": "Mixed news",
            }

    monkeypatch.setattr(graph, "get_forecast", lambda ticker: forecast)
    monkeypatch.setattr(graph, "build_graph", lambda: FakeGraph())
    monkeypatch.setattr(graph.ReportCache, "get", lambda self, ticker: None)
    monkeypatch.setattr(graph.ReportCache, "set", lambda self, ticker, result: None)

    result = graph.analyze_stock("NVDA")

    assert result["status"] == "ok"
    assert result["recommendation"] == "BULLISH"
    assert result["predictions"]["forecast"][0]["value"] == 101.0
    assert result["performance_analysis"] == "Upward bias"
    assert result["cached"] is False
