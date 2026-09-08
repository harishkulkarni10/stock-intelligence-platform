"""Deterministic agent evaluation fixtures (no Ollama / no live news APIs)."""

from __future__ import annotations

from typing import Any

from src.agents.news_guardrails import (
    build_news_facts,
    deterministic_news_analysis,
    run_news_harness,
    validate_news_analysis,
)
from src.agents.performance_guardrails import (
    build_forecast_facts,
    deterministic_performance_analysis,
    run_performance_harness,
    validate_performance_analysis,
)
from src.agents.tools import format_forecast_for_prompt, format_news_for_prompt

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
                {"step": i, "date": f"2026-09-0{i + 2}", "value": 224.41}
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

NEWS_EVAL_FIXTURES: tuple[dict[str, Any], ...] = (
    {
        "id": "news_unavailable",
        "ticker": "NVDA",
        "news": {
            "status": "error",
            "ticker": "NVDA",
            "provider": None,
            "articles": [],
            "error": "finnhub: timeout; yahoo: empty",
        },
        "expect_allowed": "UNAVAILABLE",
    },
    {
        "id": "news_positive_bundle",
        "ticker": "NVDA",
        "news": {
            "status": "ok",
            "ticker": "NVDA",
            "provider": "fixture",
            "articles": [
                {
                    "date": "2026-09-01",
                    "headline": "NVIDIA beats earnings estimates on strong GPU demand",
                    "summary": "Data center revenue surged as cloud customers expanded AI clusters.",
                    "url": "https://example.com/nvda-beats",
                },
                {
                    "date": "2026-09-02",
                    "headline": "Analysts raise NVIDIA price targets after guidance lift",
                    "summary": "Multiple firms cited sustained accelerator backlog.",
                    "url": "https://example.com/nvda-targets",
                },
            ],
        },
        "expect_allowed": "HAS_ARTICLES",
        "good_analysis": (
            "Sentiment: POSITIVE\n"
            "Drivers: - NVIDIA beats earnings estimates on strong GPU demand\n"
            "- Analysts raise NVIDIA price targets after guidance lift\n"
            "Caveat: Coverage is recent but may miss macro risks."
        ),
    },
    {
        "id": "invented_headline_must_fail",
        "ticker": "AAPL",
        "news": {
            "status": "ok",
            "ticker": "AAPL",
            "provider": "fixture",
            "articles": [
                {
                    "date": "2026-09-01",
                    "headline": "Apple unveils quieter iPhone update focused on battery life",
                    "summary": "Incremental camera and efficiency improvements.",
                    "url": "https://example.com/aapl",
                }
            ],
        },
        "expect_allowed": "HAS_ARTICLES",
        "bad_analysis": (
            "Sentiment: POSITIVE\n"
            "Drivers: - Apple acquires a secret quantum chip startup in Zurich for forty billion dollars overnight\n"
            "Caveat: None."
        ),
        "expect_validation_ok": False,
    },
)


def evaluate_performance_fixtures() -> dict[str, Any]:
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
            row["validation_pass"] = check.ok == fixture.get("expect_validation_ok", True)
            row["errors"] = list(check.errors)
        else:
            good = deterministic_performance_analysis(facts, fixture["ticker"])
            check = validate_performance_analysis(good, facts)
            row["validation_ok"] = check.ok
            row["validation_pass"] = check.ok

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
        r["trend_ok"] and r.get("validation_pass", True) and r.get("harness_recovers", True)
        for r in rows
    )
    return {"ok": passed, "cases": rows}


def evaluate_news_fixtures() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fixture in NEWS_EVAL_FIXTURES:
        news = fixture["news"]
        facts = build_news_facts(news, ticker=fixture["ticker"])
        row: dict[str, Any] = {
            "id": fixture["id"],
            "allowed_ok": facts.allowed_sentiment == fixture["expect_allowed"],
            "allowed": facts.allowed_sentiment,
        }
        if "bad_analysis" in fixture:
            check = validate_news_analysis(fixture["bad_analysis"], facts)
            row["validation_ok"] = check.ok
            row["validation_pass"] = check.ok == fixture.get("expect_validation_ok", True)
            row["errors"] = list(check.errors)
        elif "good_analysis" in fixture:
            check = validate_news_analysis(fixture["good_analysis"], facts)
            row["validation_ok"] = check.ok
            row["validation_pass"] = check.ok
        else:
            good = deterministic_news_analysis(facts)
            check = validate_news_analysis(good, facts)
            row["validation_ok"] = check.ok
            row["validation_pass"] = check.ok

            class _BadLLM:
                def invoke(self, messages):
                    return (
                        "Sentiment: POSITIVE\n"
                        "Drivers: - Completely fabricated merger with a fictional bank in Antarctica tomorrow\n"
                        "Caveat: none"
                    )

            recovered = run_news_harness(
                ticker=fixture["ticker"],
                news=news,
                news_raw=format_news_for_prompt(news),
                llm=_BadLLM(),
                max_attempts=1,
            )
            row["harness_recovers"] = (
                recovered["news_guardrail_ok"]
                and recovered["news_sentiment"] == "UNAVAILABLE"
            )
        rows.append(row)

    passed = all(
        r["allowed_ok"] and r.get("validation_pass", True) and r.get("harness_recovers", True)
        for r in rows
    )
    return {"ok": passed, "cases": rows}


def evaluate_all_agent_fixtures() -> dict[str, Any]:
    performance = evaluate_performance_fixtures()
    news = evaluate_news_fixtures()
    return {
        "ok": performance["ok"] and news["ok"],
        "performance": performance,
        "news": news,
    }


class AgentEvaluator:
    def evaluate_performance(self) -> dict[str, Any]:
        return evaluate_performance_fixtures()

    def evaluate_news(self) -> dict[str, Any]:
        return evaluate_news_fixtures()

    def evaluate_all(self) -> dict[str, Any]:
        return evaluate_all_agent_fixtures()
