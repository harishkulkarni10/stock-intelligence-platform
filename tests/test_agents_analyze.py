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
    monkeypatch.setattr(graph.ReportCache, "read", lambda self, ticker: None)

    result = graph.analyze_stock("NVDA")

    assert result["status"] == "missing_model"
    assert result["ticker"] == "NVDA"
    assert result["mode"] == "performance_news_financial"


def test_analyze_stock_happy_path_performance_news_financial(monkeypatch):
    from src.agents import graph

    forecast = {
        "status": "ok",
        "ticker": "NVDA",
        "horizon": 5,
        "model_source": "persistence",
        "model_version": "persistence",
        "model_type": "persistence",
        "last_close": 100.0,
        "last_date": "2026-08-14",
        "history": [],
        "predictions": [
            {"step": 1, "date": "2026-08-17", "value": 100.0},
            {"step": 2, "date": "2026-08-18", "value": 100.0},
            {"step": 3, "date": "2026-08-19", "value": 100.0},
            {"step": 4, "date": "2026-08-20", "value": 100.0},
            {"step": 5, "date": "2026-08-21", "value": 100.0},
        ],
        "champion": "persistence",
        "beats_persistence": False,
    }

    class FakeGraph:
        def invoke(self, state):
            return {
                "performance_analysis": (
                    "Trend: SIDEWAYS\n"
                    "Range: 100.0000 – 100.0000\n"
                    "Caution: Persistence baseline."
                ),
                "performance_trend": "SIDEWAYS",
                "performance_guardrail_ok": True,
                "performance_repaired": False,
                "news_summary": (
                    "Sentiment: MIXED\n"
                    "Drivers: - Sample coverage\n"
                    "Caveat: Fixture."
                ),
                "news_sentiment": "MIXED",
                "news_guardrail_ok": True,
                "news_repaired": False,
                "news": {"status": "ok", "provider": "fixture", "articles": []},
                "financial_analysis": (
                    "Health: ADEQUATE\n"
                    "Analysis:\nFixture fundamentals note with enough length to satisfy "
                    "UI rendering without claiming live Yahoo figures in this unit test.\n"
                    "Strengths:\n- Fixture.\n"
                    "Weaknesses:\n- Fixture.\n"
                    "Caveats: Fixture."
                ),
                "financial_health": "ADEQUATE",
                "financial_guardrail_ok": True,
                "financial_repaired": False,
                "financials": {
                    "status": "partial",
                    "ticker": "NVDA",
                    "coverage": "partial",
                    "allowed_health": "ADEQUATE",
                    "metrics": {"revenue": 1.0},
                    "signals": {"profitability": "stable", "leverage": "moderate", "valuation": "fair"},
                },
            }

    monkeypatch.setattr(graph, "get_forecast", lambda ticker: forecast)
    monkeypatch.setattr(
        graph, "build_performance_news_financial_graph", lambda: FakeGraph()
    )
    monkeypatch.setattr(graph.ReportCache, "read", lambda self, ticker: None)
    monkeypatch.setattr(graph.ReportCache, "set", lambda self, ticker, result: None)
    monkeypatch.setattr(
        graph,
        "get_company_profile",
        lambda ticker: {"ticker": ticker, "name": ticker, "status": "ok", "summary": "x"},
    )
    monkeypatch.setattr(
        graph,
        "explain_summary_chips",
        lambda **kwargs: {
            "trend": "Trend blurb for test.",
            "news": "News blurb for test.",
            "confidence": "Confidence blurb for test.",
            "projected_move": "Move blurb for test.",
        },
    )

    result = graph.analyze_stock("NVDA")

    assert result["status"] == "ok"
    assert result["mode"] == "performance_news_financial"
    assert result["performance_trend"] == "SIDEWAYS"
    assert result["recommendation"] == "NEUTRAL"
    assert result["news_sentiment"] == "MIXED"
    assert result["news_guardrail_ok"] is True
    assert result["financial_health"] == "ADEQUATE"
    assert result["financial_guardrail_ok"] is True
    assert result["financials"]["coverage"] == "partial"
    assert result["predictions"]["forecast"][0]["value"] == 100.0
    assert result["cached"] is False
    assert result["metric_explanations"]["trend"] == "Trend blurb for test."
    assert result["metric_explanations"]["news"] == "News blurb for test."
    assert result["confidence"] == "Low"  # persistence → Low
    assert result["metric_explanations"]["projected_move"] == "Move blurb for test."


def test_analyze_stock_force_refresh_skips_cache(monkeypatch):
    from src.agents import graph

    calls = {"get": 0, "delete": 0, "set": 0}

    class FakeCache:
        def __init__(self, *args, **kwargs):
            pass

        def read(self, ticker):
            calls["get"] += 1
            return {
                "result": {"status": "ok", "ticker": ticker, "performance_analysis": "cached"},
                "cached_at_ts": 1,
                "ttl_seconds": 3600,
                "age_seconds": 10,
            }

        def delete(self, ticker):
            calls["delete"] += 1

        def set(self, ticker, result):
            calls["set"] += 1

    class FakeGraph:
        def invoke(self, state):
            return {
                "performance_analysis": "Trend: SIDEWAYS\nAnalysis:\nFresh.",
                "performance_trend": "SIDEWAYS",
                "performance_guardrail_ok": True,
                "performance_repaired": False,
                "news_summary": "Sentiment: MIXED\nAnalysis:\nFresh.",
                "news_sentiment": "MIXED",
                "news_guardrail_ok": True,
                "news_repaired": False,
                "news": {},
                "financial_analysis": "Health: ADEQUATE\nAnalysis:\nFresh.",
                "financial_health": "ADEQUATE",
                "financial_guardrail_ok": True,
                "financial_repaired": False,
                "financials": {"coverage": "partial", "allowed_health": "ADEQUATE"},
            }

    forecast = {
        "status": "ok",
        "ticker": "AVGO",
        "horizon": 5,
        "model_source": "persistence",
        "predictions": [{"step": 1, "date": "2026-01-02", "value": 100.0}],
        "history": [],
        "last_close": 100.0,
        "last_date": "2026-01-01",
        "evaluation": {},
    }

    monkeypatch.setattr(graph, "ReportCache", FakeCache)
    monkeypatch.setattr(graph, "get_forecast", lambda ticker: forecast)
    monkeypatch.setattr(
        graph, "build_performance_news_financial_graph", lambda: FakeGraph()
    )
    monkeypatch.setattr(
        graph,
        "get_company_profile",
        lambda ticker: {"ticker": ticker, "name": ticker, "status": "ok"},
    )
    monkeypatch.setattr(
        graph,
        "explain_summary_chips",
        lambda **kwargs: {
            "trend": "t",
            "news": "n",
            "confidence": "c",
            "projected_move": "m",
        },
    )

    result = graph.analyze_stock("AVGO", force_refresh=True)

    assert calls["delete"] == 1
    assert calls["get"] == 0
    assert result["cached"] is False
    assert "Fresh" in (result.get("performance_analysis") or "")


def test_build_graphs_available():
    from src.agents.graph import (
        build_full_graph,
        build_performance_graph,
        build_performance_news_financial_graph,
        build_performance_news_graph,
    )

    assert build_performance_graph() is not None
    assert build_performance_news_graph() is not None
    assert build_performance_news_financial_graph() is not None
    assert build_full_graph() is not None
