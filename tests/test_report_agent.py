"""Unit tests for Report / Synthesis agent (agent 5) guardrails and harness."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agents.nodes import report_generator_node
from src.agents.report_guardrails import (
    build_report_facts,
    deterministic_report,
    run_report_harness,
    stance_from_trend,
    validate_report,
)
from src.monitoring.agent_eval import evaluate_report_fixtures


def _forecast(*, source: str = "child", last: float = 100.0, terminal: float = 110.0):
    return {
        "status": "ok",
        "ticker": "NVDA",
        "model_source": source,
        "last_close": last,
        "horizon": 5,
        "predictions": [
            {"step": 1, "date": "2026-02-01", "value": last},
            {"step": 5, "date": "2026-02-05", "value": terminal},
        ],
    }


def test_stance_from_trend_maps_sideways():
    assert stance_from_trend("SIDEWAYS") == "NEUTRAL"
    assert stance_from_trend("BULLISH") == "BULLISH"
    assert stance_from_trend("BEARISH") == "BEARISH"


def test_deterministic_report_passes_validation():
    forecast = _forecast()
    facts = build_report_facts(
        ticker="NVDA",
        forecast=forecast,
        forecast_text="last_close=100 predictions=[110]",
        performance_trend="BULLISH",
        performance_analysis="Trend: BULLISH\nAnalysis:\nUp path.",
        news_sentiment="POSITIVE",
        news_summary="Sentiment: POSITIVE\nAnalysis:\nSupportive.",
        financial_health="STRONG",
        financial_analysis="Health: STRONG\nAnalysis:\nSolid.",
        risk_level="CONTAINED",
        risk_analysis="Risk: CONTAINED\nAnalysis:\nCalm.",
        confidence="Medium",
    )
    text = deterministic_report(facts)
    result = validate_report(text, facts)
    assert result.ok is True
    assert result.parsed_stance == "BULLISH"
    assert result.parsed_confidence == "Medium"


def test_wrong_stance_fails():
    forecast = _forecast(terminal=90.0)
    facts = build_report_facts(
        ticker="NVDA",
        forecast=forecast,
        performance_trend="BEARISH",
        news_sentiment="NEGATIVE",
        financial_health="STRESSED",
        risk_level="ELEVATED",
        confidence="Medium",
    )
    bad = deterministic_report(facts).replace("Stance: BEARISH", "Stance: BULLISH", 1)
    result = validate_report(bad, facts)
    assert result.ok is False
    assert any(err.startswith("stance_mismatch") for err in result.errors)


def test_advice_language_fails():
    forecast = _forecast()
    facts = build_report_facts(
        ticker="NVDA",
        forecast=forecast,
        performance_trend="BULLISH",
        news_sentiment="MIXED",
        financial_health="ADEQUATE",
        risk_level="MODERATE",
        confidence="Medium",
    )
    bad = deterministic_report(facts) + "\nPlease buy the stock tomorrow."
    result = validate_report(bad, facts)
    assert result.ok is False
    assert "advice_language" in result.errors


def test_harness_repairs_bad_llm():
    class BadLLM:
        def invoke(self, messages):
            return AIMessage(
                content=(
                    "Stance: BULLISH\n"
                    "Confidence: High\n\n"
                    "Executive summary:\nToo short. Buy now at 999.99.\n"
                    "Caveats: none"
                )
            )

    out = run_report_harness(
        ticker="NVDA",
        forecast=_forecast(source="persistence", terminal=100.0),
        forecast_text="last_close=100 flat path",
        performance_trend="SIDEWAYS",
        news_sentiment="MIXED",
        financial_health="ADEQUATE",
        risk_level="MODERATE",
        confidence="Low",
        llm=BadLLM(),
        max_attempts=1,
    )
    assert out["report_guardrail_ok"] is True
    assert out["recommendation"] == "NEUTRAL"
    assert out["confidence"] == "Low"
    assert out["final_report"].startswith("Stance: NEUTRAL")
    assert "999.99" not in out["final_report"]
    assert "buy now" not in out["final_report"].lower()


def test_report_generator_node(monkeypatch):
    from src.agents import nodes

    class JunkLLM:
        def invoke(self, messages):
            return AIMessage(content="Stance: BULLISH\nConfidence: High\nshort")

    monkeypatch.setattr(nodes, "llm", JunkLLM())
    result = report_generator_node(
        {
            "ticker": "NVDA",
            "forecast": _forecast(source="persistence", terminal=100.0),
            "forecast_text": "flat",
            "performance_trend": "SIDEWAYS",
            "performance_repaired": False,
            "performance_analysis": "Trend: SIDEWAYS\nAnalysis:\nFlat.",
            "news_sentiment": "MIXED",
            "news_summary": "Sentiment: MIXED\nAnalysis:\nMixed.",
            "financial_health": "ADEQUATE",
            "financial_analysis": "Health: ADEQUATE\nAnalysis:\nOk.",
            "risk_level": "MODERATE",
            "risk_analysis": "Risk: MODERATE\nAnalysis:\nOk.",
            "risk": {"allowed_risk": "MODERATE"},
        }
    )
    assert result["recommendation"] == "NEUTRAL"
    assert result["confidence"] == "Low"
    assert result["report_guardrail_ok"] is True
    assert "Stance: NEUTRAL" in result["final_report"]


def test_report_eval_fixtures_pass():
    report = evaluate_report_fixtures()
    assert report["ok"] is True, report
