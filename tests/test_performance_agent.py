"""Unit tests for Performance Analyst guardrails, harness, and eval fixtures."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agents.nodes import performance_analyst_node
from src.agents.performance_guardrails import (
    build_forecast_facts,
    expected_trend_from_prices,
    parse_trend_label,
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
    assert "999" not in out["performance_analysis"]


def test_performance_analyst_node_uses_harness(monkeypatch):
    from src.agents import nodes

    class GoodLLM:
        def invoke(self, messages):
            return AIMessage(
                content=(
                    "Trend: SIDEWAYS\n"
                    "Range: 224.4100 – 224.4100\n"
                    "Caution: Persistence baseline; limited signal."
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


def test_performance_eval_fixtures_pass():
    report = evaluate_performance_fixtures()
    assert report["ok"] is True, report
