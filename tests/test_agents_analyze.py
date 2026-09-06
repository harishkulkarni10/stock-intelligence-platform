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
    assert result["mode"] == "performance_only"


def test_analyze_stock_happy_path_agent1_only(monkeypatch):
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
            }

    monkeypatch.setattr(graph, "get_forecast", lambda ticker: forecast)
    monkeypatch.setattr(graph, "build_performance_graph", lambda: FakeGraph())
    monkeypatch.setattr(graph.ReportCache, "get", lambda self, ticker: None)
    monkeypatch.setattr(graph.ReportCache, "set", lambda self, ticker, result: None)

    result = graph.analyze_stock("NVDA")

    assert result["status"] == "ok"
    assert result["mode"] == "performance_only"
    assert result["performance_trend"] == "SIDEWAYS"
    assert result["recommendation"] == "NEUTRAL"
    assert result["confidence"] == "Low"
    assert result["performance_guardrail_ok"] is True
    assert result["news_summary"] is None
    assert result["draft_report"] is None
    assert result["predictions"]["forecast"][0]["value"] == 100.0
    assert "SIDEWAYS" in (result["performance_analysis"] or "")
    assert result["cached"] is False


def test_build_graphs_available():
    from src.agents.graph import build_full_graph, build_performance_graph

    assert build_performance_graph() is not None
    assert build_full_graph() is not None
