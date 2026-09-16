"""Unit tests for Financial Analyst (agent 3) guardrails and harness."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agents.financial_guardrails import (
    build_financial_facts,
    deterministic_financial_analysis,
    run_financial_harness,
    validate_financial_analysis,
)
from src.agents.nodes import financial_analyst_node
from src.market.financials import (
    derive_allowed_health,
    format_financials_for_prompt,
)
from src.monitoring.agent_eval import evaluate_financial_fixtures


def _strong_financials() -> dict:
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
        "ticker": "NVDA",
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
        "allowed_numbers": [float(v) for v in metrics.values()],
    }


def test_missing_financials_force_unavailable():
    facts = build_financial_facts(
        {
            "status": "missing",
            "ticker": "ZZZZ",
            "coverage": "missing",
            "allowed_health": "UNAVAILABLE",
            "metrics": {},
            "signals": {},
            "allowed_numbers": [],
        },
        ticker="ZZZZ",
    )
    assert facts.allowed_health == "UNAVAILABLE"
    bad = "Health: STRONG\nAnalysis:\nRevenue is wonderful.\nStrengths:\n- x\nWeaknesses:\n- y\nCaveats: z"
    assert validate_financial_analysis(bad, facts).ok is False


def test_deterministic_strong_passes():
    financials = _strong_financials()
    facts = build_financial_facts(financials)
    text = deterministic_financial_analysis(facts)
    result = validate_financial_analysis(text, facts)
    assert result.ok is True
    assert result.parsed_health == "STRONG"


def test_invented_revenue_fails():
    financials = _strong_financials()
    facts = build_financial_facts(financials)
    text = (
        "Health: STRONG\n"
        "Analysis:\n"
        "The firm suddenly booked 999.0B of brand-new revenue that is nowhere in the "
        "fetched metrics, which should fail grounding because that magnitude is not "
        "near any allowed fundamental number from the snapshot provided to the agent.\n"
        "Strengths:\n- Imaginary growth.\n"
        "Weaknesses:\n- None.\n"
        "Caveats: none"
    )
    assert validate_financial_analysis(text, facts).ok is False


def test_wrong_health_fails():
    financials = _strong_financials()
    facts = build_financial_facts(financials)
    text = deterministic_financial_analysis(facts).replace("Health: STRONG", "Health: STRESSED")
    assert validate_financial_analysis(text, facts).ok is False


def test_harness_recovers_when_llm_invents():
    financials = {
        "status": "missing",
        "ticker": "MSFT",
        "coverage": "missing",
        "allowed_health": "UNAVAILABLE",
        "metrics": {},
        "signals": {},
        "allowed_numbers": [],
    }

    class BadLLM:
        def invoke(self, messages):
            return AIMessage(
                content=(
                    "Health: STRONG\n"
                    "Analysis:\nCash is infinite.\n"
                    "Strengths:\n- x\nWeaknesses:\n- y\nCaveats: z"
                )
            )

    out = run_financial_harness(
        ticker="MSFT",
        financials=financials,
        financials_raw=format_financials_for_prompt(financials),
        llm=BadLLM(),
        max_attempts=1,
    )
    assert out["financial_health"] == "UNAVAILABLE"
    assert out["financial_guardrail_ok"] is True
    assert out["financial_analysis"].startswith("Health: UNAVAILABLE")


def test_derive_stressed_on_weak_cash():
    health = derive_allowed_health(
        coverage="full",
        profitability_signal="weak",
        leverage_signal="stretched",
        fcf=-1_000_000.0,
        net_margin_pct=-2.0,
    )
    assert health == "STRESSED"


def test_financial_analyst_node_uses_harness(monkeypatch):
    from src.agents import nodes

    financials = _strong_financials()

    class GoodLLM:
        def invoke(self, messages):
            return AIMessage(content=deterministic_financial_analysis(build_financial_facts(financials)))

    monkeypatch.setattr(nodes, "llm", GoodLLM())
    monkeypatch.setattr(nodes, "get_financials", lambda ticker: financials)

    out = financial_analyst_node({"ticker": "NVDA"})
    assert out["financial_health"] == "STRONG"
    assert out["financial_guardrail_ok"] is True
    assert "Health: STRONG" in out["financial_analysis"]


def test_financial_eval_fixtures_pass():
    report = evaluate_financial_fixtures()
    assert report["ok"] is True
