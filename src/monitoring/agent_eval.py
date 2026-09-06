"""Agent evaluation helpers.

Performance Analyst fixtures are deterministic (no Ollama) so CI can gate quality.
"""

from __future__ import annotations

from typing import Any

from src.agents.performance_guardrails import (
    build_forecast_facts,
    deterministic_performance_analysis,
    run_performance_harness,
    validate_performance_analysis,
)
from src.agents.tools import format_forecast_for_prompt

# Golden cases: structured forecast → required trend. Used as the eval set.
PERFORMANCE_EVAL_FIXTURES: tuple[dict[str, Any], ...] = (
    {
        "id": "flat_persistence",
        "ticker": "NVDA",
        "forecast": {
            "status": "ok",
            "ticker": "NVDA",
            "last_close": 224.41,
            "last_date": "2026-09-02",
            "model_source": "persistence",
            "model_version": "persistence",
            "predictions": [
                {"step": i, "date": f"2026-09-0{i+2}", "value": 224.41}
                for i in range(1, 6)
            ],
        },
        "expected_trend": "SIDEWAYS",
    },
    {
        "id": "clear_up",
        "ticker": "AAPL",
        "forecast": {
            "status": "ok",
            "ticker": "AAPL",
            "last_close": 100.0,
            "last_date": "2026-01-01",
            "model_source": "child",
            "model_version": "test",
            "predictions": [
                {"step": 1, "date": "2026-01-02", "value": 101.0},
                {"step": 2, "date": "2026-01-03", "value": 103.0},
                {"step": 3, "date": "2026-01-04", "value": 105.0},
                {"step": 4, "date": "2026-01-05", "value": 107.0},
                {"step": 5, "date": "2026-01-06", "value": 110.0},
            ],
        },
        "expected_trend": "BULLISH",
    },
    {
        "id": "clear_down",
        "ticker": "MSFT",
        "forecast": {
            "status": "ok",
            "ticker": "MSFT",
            "last_close": 400.0,
            "last_date": "2026-01-01",
            "model_source": "parent",
            "model_version": "test",
            "predictions": [
                {"step": 1, "date": "2026-01-02", "value": 395.0},
                {"step": 2, "date": "2026-01-03", "value": 390.0},
                {"step": 3, "date": "2026-01-04", "value": 385.0},
                {"step": 4, "date": "2026-01-05", "value": 380.0},
                {"step": 5, "date": "2026-01-06", "value": 370.0},
            ],
        },
        "expected_trend": "BEARISH",
    },
    {
        "id": "invented_price_must_fail",
        "ticker": "NVDA",
        "forecast": {
            "status": "ok",
            "ticker": "NVDA",
            "last_close": 100.0,
            "predictions": [
                {"step": 1, "value": 100.0},
                {"step": 2, "value": 100.5},
                {"step": 3, "value": 101.0},
                {"step": 4, "value": 101.5},
                {"step": 5, "value": 102.0},
            ],
        },
        "expected_trend": "BULLISH",
        "bad_analysis": (
            "Trend: BULLISH\n"
            "Range: 100.00 – 150.00\n"
            "Caution: Target looks like 150.00."
        ),
        "expect_validation_ok": False,
    },
)


def evaluate_performance_fixtures() -> dict[str, Any]:
    """Score the Performance Analyst policy layer without calling an LLM."""
    rows: list[dict[str, Any]] = []
    for fixture in PERFORMANCE_EVAL_FIXTURES:
        forecast = fixture["forecast"]
        facts = build_forecast_facts(forecast)
        trend_ok = facts.expected_trend == fixture["expected_trend"]
        row: dict[str, Any] = {
            "id": fixture["id"],
            "trend_ok": trend_ok,
            "expected_trend": fixture["expected_trend"],
            "got_trend": facts.expected_trend,
        }
        if "bad_analysis" in fixture:
            check = validate_performance_analysis(fixture["bad_analysis"], facts)
            row["validation_ok"] = check.ok
            row["validation_pass"] = check.ok == fixture.get(
                "expect_validation_ok", True
            )
            row["errors"] = list(check.errors)
        else:
            good = deterministic_performance_analysis(facts, fixture["ticker"])
            check = validate_performance_analysis(good, facts)
            row["validation_ok"] = check.ok
            row["validation_pass"] = check.ok
            # Harness must recover from a wrong LLM answer.
            class _BadLLM:
                def invoke(self, messages):
                    return (
                        "Trend: BULLISH\n"
                        "Range: 999.99 – 1000.00\n"
                        "Caution: ignoring the forecast."
                    )

            recovered = run_performance_harness(
                ticker=fixture["ticker"],
                forecast_text=format_forecast_for_prompt(forecast),
                forecast=forecast,
                llm=_BadLLM(),
                max_attempts=1,
            )
            row["harness_recovers"] = (
                recovered["performance_guardrail_ok"]
                and recovered["performance_trend"] == fixture["expected_trend"]
                and "999.99" not in recovered["performance_analysis"]
            )
        rows.append(row)

    passed = all(
        r["trend_ok"]
        and r.get("validation_pass", True)
        and r.get("harness_recovers", True)
        for r in rows
    )
    return {"ok": passed, "cases": rows}


class AgentEvaluator:
    """Thin entrypoint; start with Performance Analyst fixtures."""

    def evaluate_performance(self) -> dict[str, Any]:
        return evaluate_performance_fixtures()
