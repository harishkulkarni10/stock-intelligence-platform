"""Unit tests for Performance Analyst guardrails, harness, and eval fixtures."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agents.nodes import performance_analyst_node
from src.agents.performance_guardrails import (
    build_forecast_facts,
    deterministic_chip_explanations,
    expected_trend_from_prices,
    explain_summary_chips,
    format_projected_move,
    parse_chip_explanations,
    parse_trend_label,
    projected_move_pct,
    run_performance_harness,
    validate_performance_analysis,
)
from src.agents.tools import format_forecast_for_prompt
from src.monitoring.agent_eval import evaluate_performance_fixtures


def _flat_forecast(price: float = 224.41) -> dict:
    return {
        "status": "ok",
        "ticker": "NVDA",
        "last_close": price,
        "last_date": "2026-09-02",
        "model_source": "persistence",
        "model_version": "persistence",
        "predictions": [
            {"step": i, "date": f"2026-09-0{i}", "value": price} for i in range(1, 6)
        ],
    }


def test_expected_trend_sideways_up_down():
    assert expected_trend_from_prices([100.0, 100.0, 100.0], last_close=100.0) == "SIDEWAYS"
    assert expected_trend_from_prices([101.0, 105.0, 110.0], last_close=100.0) == "BULLISH"
    assert expected_trend_from_prices([99.0, 95.0, 90.0], last_close=100.0) == "BEARISH"


def test_flat_persistence_facts_are_sideways():
    facts = build_forecast_facts(_flat_forecast())
    assert facts.expected_trend == "SIDEWAYS"
    assert facts.low == facts.high == 224.41


def test_validate_rejects_bullish_on_flat():
    facts = build_forecast_facts(_flat_forecast())
    bad = (
        "Based on the persistence model the projected trend appears to be Bullish, "
        "as predicted closes remain unchanged at $224.41."
    )
    result = validate_performance_analysis(bad, facts)
    assert result.ok is False
    assert any(err.startswith("trend_mismatch") for err in result.errors)


def test_validate_rejects_invented_price():
    facts = build_forecast_facts(_flat_forecast())
    bad = "Trend: SIDEWAYS\nRange: 224.41 – 999.99\nCaution: stretched target."
    result = validate_performance_analysis(bad, facts)
    assert result.ok is False
    assert any(err.startswith("invented_price") for err in result.errors)


def test_parse_trend_prefers_structured_line():
    text = "Trend: SIDEWAYS\nRange: 1.00 – 1.00\nSome bullish wording should not win."
    assert parse_trend_label(text) == "SIDEWAYS"


def test_harness_repairs_bad_llm(monkeypatch):
    class BadThenIgnored:
        def invoke(self, messages):
            return AIMessage(
                content=(
                    "Trend: BULLISH\n"
                    "Range: 224.41 – 224.41\n"
                    "Caution: holds steady so bullish."
                )
            )

    forecast = _flat_forecast()
    out = run_performance_harness(
        ticker="NVDA",
        forecast_text=format_forecast_for_prompt(forecast),
        forecast=forecast,
        llm=BadThenIgnored(),
        max_attempts=1,
    )
    assert out["performance_trend"] == "SIDEWAYS"
    assert out["performance_guardrail_ok"] is True
    assert out["performance_analysis"].startswith("Trend: SIDEWAYS")
    assert "Analysis:" in out["performance_analysis"]
    assert "999" not in out["performance_analysis"]


def test_performance_analyst_node_uses_harness(monkeypatch):
    from src.agents import nodes

    class GoodLLM:
        def invoke(self, messages):
            return AIMessage(
                content=(
                    "Trend: SIDEWAYS\n"
                    "Range: 224.4100 – 224.4100\n"
                    "Analysis:\n"
                    "The printed sessions for NVDA sit flat at 224.4100 across the horizon, "
                    "so the path does not extend away from the last close in a meaningful way. "
                    "That geometry is classic consolidation: the model is restating the current "
                    "level rather than projecting a decisive climb or slide. A careful reader "
                    "should treat the SIDEWAYS label as earned by the near-zero move and give "
                    "the note less directional conviction than a staircase of higher or lower "
                    "closes. Short-horizon flat paths are still useful for framing the chart — "
                    "they say the system sees limited signal — but they are not a claim that "
                    "the live market will stay frozen. Use the span and session list beside the "
                    "chart, and remember news shocks can invalidate a quiet baseline quickly.\n"
                    "Key points:\n"
                    "- Flat sessions justify SIDEWAYS rather than forced bullish or bearish labels.\n"
                    "- The range collapses to a single printed level, underscoring low path volatility.\n"
                    "- Treat this as a low-conviction research framing, not a trade call.\n"
                    "Caveats: Persistence-like flats miss regime shifts; research support only."
                )
            )

    monkeypatch.setattr(nodes, "llm", GoodLLM())
    forecast = _flat_forecast()
    result = performance_analyst_node(
        {
            "ticker": "NVDA",
            "forecast": forecast,
            "forecast_text": format_forecast_for_prompt(forecast),
        }
    )
    assert result["performance_trend"] == "SIDEWAYS"
    assert result["performance_guardrail_ok"] is True
    assert "SIDEWAYS" in result["performance_analysis"]
    assert "Analysis:" in result["performance_analysis"]


def test_performance_eval_fixtures_pass():
    report = evaluate_performance_fixtures()
    assert report["ok"] is True, report


def test_projected_move_pct_and_format():
    forecast = {
        "last_close": 100.0,
        "predictions": [{"value": 101.5}],
    }
    assert abs(projected_move_pct(forecast) - 1.5) < 1e-9
    assert format_projected_move(1.5) == "+1.50%"
    assert format_projected_move(None) == "—"


def test_parse_chip_explanations():
    text = (
        "Trend:\nBullish because the path rises.\n\n"
        "News:\nMixed headlines.\n\n"
        "Confidence:\nMedium quality path.\n\n"
        "Projected move:\nAbout +1.50% over five sessions.\n"
    )
    parsed = parse_chip_explanations(text)
    assert "Bullish" in parsed["trend"]
    assert "Mixed" in parsed["news"]
    assert "Medium" in parsed["confidence"]
    assert "+1.50%" in parsed["projected_move"]


def test_explain_summary_chips_uses_llm_and_fallback():
    class Scripted:
        def invoke(self, messages):
            return AIMessage(
                content=(
                    "Trend:\nPath ends higher so Trend is BULLISH for NVDA.\n"
                    "News:\nBriefing tone is MIXED across drivers.\n"
                    "Confidence:\nLow because this run used a simpler path.\n"
                    "Projected move:\nMove is +0.00% from last close to the final session.\n"
                )
            )

    out = explain_summary_chips(
        ticker="NVDA",
        trend="BULLISH",
        news_sentiment="MIXED",
        confidence="Low",
        forecast=_flat_forecast(),
        performance_analysis="Trend: BULLISH\nAnalysis:\nUp path.",
        news_summary="Sentiment: MIXED\nAnalysis:\nMixed.",
        llm=Scripted(),
    )
    assert "BULLISH" in out["trend"]
    assert "MIXED" in out["news"]
    assert out["confidence"]
    assert out["projected_move"]

    fallback = explain_summary_chips(
        ticker="NVDA",
        trend="SIDEWAYS",
        news_sentiment="MIXED",
        confidence="Low",
        forecast=_flat_forecast(),
    )
    assert set(fallback) == {"trend", "news", "confidence", "projected_move", "risk"}
    assert "SIDEWAYS" in fallback["trend"]

    det = deterministic_chip_explanations(
        ticker="CCL",
        trend="BULLISH",
        news_sentiment="MIXED",
        confidence="Medium",
        projected_move="+1.48%",
        horizon=5,
    )
    assert "BULLISH" in det["trend"]
    assert "+1.48%" in det["projected_move"]
