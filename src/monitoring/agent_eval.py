"""Deterministic agent evaluation fixtures (no Ollama / no live news APIs)."""

from __future__ import annotations

from typing import Any

from src.agents.news_guardrails import (
    build_news_facts,
    deterministic_news_analysis,
    run_news_harness,
    validate_news_analysis,
)
from src.agents.financial_guardrails import (
    build_financial_facts,
    deterministic_financial_analysis,
    run_financial_harness,
    validate_financial_analysis,
)
from src.agents.performance_guardrails import (
    build_forecast_facts,
    deterministic_performance_analysis,
    run_performance_harness,
    validate_performance_analysis,
)
from src.agents.report_guardrails import (
    build_report_facts,
    deterministic_report,
    run_report_harness,
    stance_from_trend,
    validate_report,
)
from src.agents.tools import (
    format_financials_for_prompt,
    format_forecast_for_prompt,
    format_news_for_prompt,
)

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
            "Headlines:\n"
            "- NVIDIA beats earnings estimates on strong GPU demand\n"
            "- Analysts raise NVIDIA price targets after guidance lift\n"
            "Analysis:\n"
            "Coverage around NVIDIA is constructive and internally consistent. The earnings "
            "beat tied to GPU demand and the follow-on lift in analyst targets both point to "
            "the same underlying story: customers are still expanding AI infrastructure and "
            "the Street is updating expectations accordingly. That does not remove execution "
            "or cyclical risk, but for a short-horizon news read the tape is supportive rather "
            "than conflicted. A careful reader should still open the primary articles, check "
            "how much of the beat was data-center versus other segments, and note whether "
            "guidance strength is broad or concentrated. Taken together, the headlines justify "
            "a POSITIVE sentiment label while leaving room for macro or competitor surprises "
            "that are not spelled out in this small sample.\n"
            "Implications:\n"
            "- Demand and guidance tone currently lean supportive for NVIDIA.\n"
            "- Target raises reinforce Street confidence but are still opinions, not facts.\n"
            "Caveats: Coverage is recent but may miss macro risks."
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


def _healthy_financials(ticker: str = "NVDA") -> dict[str, Any]:
    metrics = {
        "revenue": 130_000_000_000.0,
        "revenue_yoy_pct": 55.0,
        "gross_margin_pct": 75.0,
        "operating_margin_pct": 55.0,
        "net_margin_pct": 50.0,
        "operating_cashflow": 50_000_000_000.0,
        "free_cashflow": 40_000_000_000.0,
        "total_cash": 30_000_000_000.0,
        "total_debt": 10_000_000_000.0,
        "debt_to_equity": 25.0,
        "current_ratio": 3.5,
        "roe_pct": 90.0,
        "roa_pct": 40.0,
        "market_cap": 3_000_000_000_000.0,
        "pe": 45.0,
        "ps": 25.0,
        "pb": 40.0,
        "ev_ebitda": 35.0,
    }
    return {
        "status": "ok",
        "ticker": ticker,
        "name": "NVIDIA Corporation",
        "sector": "Technology",
        "industry": "Semiconductors",
        "currency": "USD",
        "coverage": "full",
        "source": "fixture",
        "metrics": metrics,
        "signals": {
            "profitability": "improving",
            "leverage": "conservative",
            "valuation": "rich",
        },
        "allowed_health": "STRONG",
        "allowed_numbers": [
            float(v) for v in metrics.values() if isinstance(v, (int, float))
        ],
    }


FINANCIAL_EVAL_FIXTURES: tuple[dict[str, Any], ...] = (
    {
        "id": "financials_unavailable",
        "ticker": "ZZZZ",
        "financials": {
            "status": "missing",
            "ticker": "ZZZZ",
            "coverage": "missing",
            "allowed_health": "UNAVAILABLE",
            "metrics": {},
            "signals": {},
            "allowed_numbers": [],
        },
        "expect_health": "UNAVAILABLE",
    },
    {
        "id": "financials_strong_bundle",
        "ticker": "NVDA",
        "financials": _healthy_financials("NVDA"),
        "expect_health": "STRONG",
    },
    {
        "id": "invented_revenue_must_fail",
        "ticker": "NVDA",
        "financials": _healthy_financials("NVDA"),
        "expect_health": "STRONG",
        "bad_analysis": (
            "Health: STRONG\n"
            "Analysis:\n"
            "The company secretly printed 999.0B of brand-new revenue overnight while "
            "keeping every other line item unchanged, which is an impossible jump versus "
            "the fetched snapshot and must be rejected by grounding checks for safety.\n"
            "Strengths:\n- Fake boom.\n"
            "Weaknesses:\n- None.\n"
            "Caveats: None."
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


def evaluate_financial_fixtures() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fixture in FINANCIAL_EVAL_FIXTURES:
        financials = fixture["financials"]
        facts = build_financial_facts(financials, ticker=fixture["ticker"])
        row: dict[str, Any] = {
            "id": fixture["id"],
            "health_ok": facts.allowed_health == fixture["expect_health"],
            "allowed_health": facts.allowed_health,
        }
        if "bad_analysis" in fixture:
            check = validate_financial_analysis(fixture["bad_analysis"], facts)
            row["validation_ok"] = check.ok
            row["validation_pass"] = check.ok == fixture.get("expect_validation_ok", True)
            row["errors"] = list(check.errors)
        else:
            good = deterministic_financial_analysis(facts)
            check = validate_financial_analysis(good, facts)
            row["validation_ok"] = check.ok
            row["validation_pass"] = check.ok

            class _BadLLM:
                def invoke(self, messages):
                    return (
                        "Health: STRONG\n"
                        "Analysis:\n"
                        "Revenue is exactly 777.77B from an undisclosed moon colony contract "
                        "that does not appear in the fundamentals block at all.\n"
                        "Strengths:\n- Imaginary contract.\n"
                        "Weaknesses:\n- None.\n"
                        "Caveats: none"
                    )

            recovered = run_financial_harness(
                ticker=fixture["ticker"],
                financials=financials,
                financials_raw=format_financials_for_prompt(financials),
                llm=_BadLLM(),
                max_attempts=1,
            )
            row["harness_recovers"] = (
                recovered["financial_guardrail_ok"]
                and recovered["financial_health"] == fixture["expect_health"]
                and "777.77" not in recovered["financial_analysis"]
            )
        rows.append(row)

    passed = all(
        r["health_ok"] and r.get("validation_pass", True) and r.get("harness_recovers", True)
        for r in rows
    )
    return {"ok": passed, "cases": rows}


REPORT_EVAL_FIXTURES: tuple[dict[str, Any], ...] = (
    {
        "id": "report_neutral_low",
        "ticker": "NVDA",
        "forecast": {
            "status": "ok",
            "ticker": "NVDA",
            "last_close": 100.0,
            "model_source": "persistence",
            "predictions": [
                {"step": 1, "value": 100.0},
                {"step": 2, "value": 100.0},
                {"step": 3, "value": 100.0},
                {"step": 4, "value": 100.0},
                {"step": 5, "value": 100.0},
            ],
        },
        "performance_trend": "SIDEWAYS",
        "news_sentiment": "MIXED",
        "financial_health": "ADEQUATE",
        "risk_level": "MODERATE",
        "confidence": "Low",
        "expected_stance": "NEUTRAL",
    },
    {
        "id": "report_bullish_medium",
        "ticker": "AAPL",
        "forecast": {
            "status": "ok",
            "ticker": "AAPL",
            "last_close": 100.0,
            "model_source": "child",
            "predictions": [
                {"step": 1, "value": 101.0},
                {"step": 2, "value": 103.0},
                {"step": 3, "value": 105.0},
                {"step": 4, "value": 107.0},
                {"step": 5, "value": 110.0},
            ],
        },
        "performance_trend": "BULLISH",
        "news_sentiment": "POSITIVE",
        "financial_health": "STRONG",
        "risk_level": "CONTAINED",
        "confidence": "Medium",
        "expected_stance": "BULLISH",
    },
    {
        "id": "invented_price_must_fail",
        "ticker": "MSFT",
        "forecast": {
            "status": "ok",
            "ticker": "MSFT",
            "last_close": 400.0,
            "model_source": "parent",
            "predictions": [
                {"step": 1, "value": 395.0},
                {"step": 5, "value": 370.0},
            ],
        },
        "performance_trend": "BEARISH",
        "news_sentiment": "NEGATIVE",
        "financial_health": "STRESSED",
        "risk_level": "ELEVATED",
        "confidence": "Medium",
        "expected_stance": "BEARISH",
        "bad_analysis": (
            "Stance: BEARISH\n"
            "Confidence: Medium\n\n"
            "Executive summary:\n"
            "MSFT looks weak with BEARISH path, NEGATIVE news, STRESSED health, "
            "ELEVATED risk, but somehow targets an invented 999.99 print that is not "
            "in the forecast packet at all and must be rejected by grounding.\n\n"
            "Forecast / performance:\nPath is BEARISH.\n\n"
            "News:\nTone NEGATIVE.\n\n"
            "Fundamentals:\nHealth STRESSED.\n\n"
            "Risk:\nLabel ELEVATED.\n\n"
            "Bull case:\n- None.\n\n"
            "Bear case:\n- Path lower.\n\n"
            "Key drivers:\n- Path.\n\n"
            "Key risks:\n- ELEVATED.\n\n"
            "Caveats: Research only."
        ),
        "expect_validation_ok": False,
    },
)


def evaluate_report_fixtures() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fixture in REPORT_EVAL_FIXTURES:
        forecast = fixture["forecast"]
        facts = build_report_facts(
            ticker=fixture["ticker"],
            forecast=forecast,
            forecast_text=format_forecast_for_prompt(forecast),
            performance_trend=fixture["performance_trend"],
            performance_analysis=f"Trend: {fixture['performance_trend']}\nAnalysis:\nFixture.",
            news_sentiment=fixture["news_sentiment"],
            news_summary=f"Sentiment: {fixture['news_sentiment']}\nAnalysis:\nFixture.",
            financial_health=fixture["financial_health"],
            financial_analysis=f"Health: {fixture['financial_health']}\nAnalysis:\nFixture.",
            risk_level=fixture["risk_level"],
            risk_analysis=f"Risk: {fixture['risk_level']}\nAnalysis:\nFixture.",
            confidence=fixture["confidence"],
        )
        stance_ok = facts.stance == fixture["expected_stance"]
        stance_ok = stance_ok and stance_from_trend(fixture["performance_trend"]) == fixture[
            "expected_stance"
        ]
        row: dict[str, Any] = {
            "id": fixture["id"],
            "stance_ok": stance_ok,
            "expected_stance": fixture["expected_stance"],
            "got_stance": facts.stance,
            "confidence": facts.confidence,
        }
        if "bad_analysis" in fixture:
            check = validate_report(fixture["bad_analysis"], facts)
            row["validation_ok"] = check.ok
            row["validation_pass"] = check.ok == fixture.get("expect_validation_ok", True)
            row["errors"] = list(check.errors)
        else:
            good = deterministic_report(facts)
            check = validate_report(good, facts)
            row["validation_ok"] = check.ok
            row["validation_pass"] = check.ok

            class _BadLLM:
                def invoke(self, messages):
                    return (
                        "Stance: BULLISH\n"
                        "Confidence: High\n\n"
                        "Executive summary:\n"
                        "Buy now at 999.99 regardless of the packet.\n\n"
                        "Caveats: none"
                    )

            recovered = run_report_harness(
                ticker=fixture["ticker"],
                forecast=forecast,
                forecast_text=format_forecast_for_prompt(forecast),
                performance_trend=fixture["performance_trend"],
                news_sentiment=fixture["news_sentiment"],
                financial_health=fixture["financial_health"],
                risk_level=fixture["risk_level"],
                confidence=fixture["confidence"],
                llm=_BadLLM(),
                max_attempts=1,
            )
            row["harness_recovers"] = (
                recovered["report_guardrail_ok"]
                and recovered["recommendation"] == fixture["expected_stance"]
                and recovered["confidence"] == fixture["confidence"]
                and "999.99" not in recovered["final_report"]
                and "buy now" not in recovered["final_report"].lower()
            )
        rows.append(row)

    passed = all(
        r["stance_ok"] and r.get("validation_pass", True) and r.get("harness_recovers", True)
        for r in rows
    )
    return {"ok": passed, "cases": rows}


def evaluate_all_agent_fixtures() -> dict[str, Any]:
    performance = evaluate_performance_fixtures()
    news = evaluate_news_fixtures()
    financial = evaluate_financial_fixtures()
    report = evaluate_report_fixtures()
    return {
        "ok": performance["ok"] and news["ok"] and financial["ok"] and report["ok"],
        "performance": performance,
        "news": news,
        "financial": financial,
        "report": report,
    }


class AgentEvaluator:
    def evaluate_performance(self) -> dict[str, Any]:
        return evaluate_performance_fixtures()

    def evaluate_news(self) -> dict[str, Any]:
        return evaluate_news_fixtures()

    def evaluate_financial(self) -> dict[str, Any]:
        return evaluate_financial_fixtures()

    def evaluate_report(self) -> dict[str, Any]:
        return evaluate_report_fixtures()

    def evaluate_all(self) -> dict[str, Any]:
        return evaluate_all_agent_fixtures()
