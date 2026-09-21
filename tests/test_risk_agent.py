"""Unit tests for Risk Analyst (agent 4) metrics, guardrails, and harness."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agents.nodes import risk_analyst_node
from src.agents.risk_guardrails import (
    build_risk_facts,
    deterministic_risk_analysis,
    run_risk_harness,
    validate_risk_analysis,
)
from src.market.risk_metrics import build_risk_metrics, derive_allowed_risk


def _history(n: int = 40, start: float = 100.0, step: float = 0.0) -> list[dict]:
    """Synthetic closes with a little noise so vol is well-defined."""
    out = []
    price = start
    for i in range(n):
        # Mild oscillation so realized vol is not ~0.
        wobble = ((i % 5) - 2) * 0.35
        out.append({"date": f"2026-01-{(i % 28) + 1:02d}", "value": round(price + wobble, 4)})
        price += step
    return out


def _forecast(*, source: str = "child", history=None, last: float = 100.0, terminal: float = 101.0):
    return {
        "status": "ok",
        "ticker": "NVDA",
        "model_source": source,
        "last_close": last,
        "horizon": 5,
        "history": history if history is not None else _history(40, start=last, step=0.05),
        "predictions": [
            {"step": 1, "date": "2026-02-01", "value": last},
            {"step": 5, "date": "2026-02-05", "value": terminal},
        ],
    }


def test_derive_risk_elevated_on_persistence_and_negative_news():
    label = derive_allowed_risk(
        model_source="persistence",
        performance_repaired=True,
        vol_ann_20d=0.55,
        max_dd=-0.35,
        move_vol_ratio=3.5,
        financial_health="STRESSED",
        leverage_signal="stretched",
        news_sentiment="NEGATIVE",
    )
    assert label == "ELEVATED"


def test_derive_risk_contained_when_calm():
    label = derive_allowed_risk(
        model_source="child",
        performance_repaired=False,
        vol_ann_20d=0.15,
        max_dd=-0.05,
        move_vol_ratio=0.4,
        financial_health="STRONG",
        leverage_signal="conservative",
        news_sentiment="POSITIVE",
    )
    assert label == "CONTAINED"


def test_build_risk_metrics_sets_allowed_label():
    risk = build_risk_metrics(
        ticker="NVDA",
        forecast=_forecast(source="persistence", terminal=100.0),
        performance_trend="SIDEWAYS",
        performance_repaired=False,
        news_sentiment="MIXED",
        financial_health="ADEQUATE",
        financials={"signals": {"leverage": "moderate"}},
    )
    assert risk["allowed_risk"] in {"CONTAINED", "MODERATE", "ELEVATED"}
    assert risk["ticker"] == "NVDA"
    assert "metrics" in risk


def test_deterministic_risk_passes_validation():
    risk = build_risk_metrics(
        ticker="NVDA",
        forecast=_forecast(),
        performance_trend="BULLISH",
        news_sentiment="MIXED",
        financial_health="ADEQUATE",
        financials={"signals": {"leverage": "conservative"}},
    )
    facts = build_risk_facts(risk)
    text = deterministic_risk_analysis(facts)
    result = validate_risk_analysis(text, facts)
    assert result.ok is True
    assert result.parsed_risk == facts.allowed_risk


def test_wrong_risk_label_fails():
    risk = build_risk_metrics(
        ticker="NVDA",
        forecast=_forecast(source="persistence"),
        news_sentiment="NEGATIVE",
        financial_health="STRESSED",
        financials={"signals": {"leverage": "stretched"}},
    )
    facts = build_risk_facts(risk)
    bad = (
        f"Risk: CONTAINED\n"
        f"Analysis:\n"
        f"{'x' * 180}\n"
        "Market / path risks:\n- a\n"
        "Financial risks:\n- b\n"
        "News / event risks:\n- c\n"
        "Forecast / model risks:\n- d\n"
        "Key downside scenarios:\n- e\n"
        "Caveats: none"
    )
    # Force mismatch regardless of allowed
    bad = bad.replace(f"Risk: CONTAINED", "Risk: CONTAINED", 1)
    if facts.allowed_risk == "CONTAINED":
        bad = bad.replace("Risk: CONTAINED", "Risk: ELEVATED", 1)
    result = validate_risk_analysis(bad, facts)
    assert result.ok is False
    assert any(err.startswith("risk_mismatch") for err in result.errors)


def test_harness_repairs_bad_llm():
    class BadLLM:
        def invoke(self, messages):
            return AIMessage(
                content=(
                    "Risk: CONTAINED\n"
                    "Analysis:\nToo short.\n"
                    "Caveats: buy now"
                )
            )

    out = run_risk_harness(
        ticker="NVDA",
        forecast=_forecast(source="persistence"),
        performance_trend="SIDEWAYS",
        news_sentiment="NEGATIVE",
        financial_health="STRESSED",
        financials={"signals": {"leverage": "stretched"}},
        llm=BadLLM(),
        max_attempts=1,
    )
    assert out["risk_guardrail_ok"] is True
    assert out["risk_level"] in {"CONTAINED", "MODERATE", "ELEVATED"}
    assert out["risk_analysis"].startswith(f"Risk: {out['risk_level']}")
    assert "buy now" not in out["risk_analysis"].lower()


def test_risk_analyst_node(monkeypatch):
    from src.agents import nodes

    class GoodLLM:
        def invoke(self, messages):
            # Let harness repair if needed — return junk then rely on repair/fallback
            return AIMessage(content="Risk: MODERATE\nAnalysis:\nshort")

    monkeypatch.setattr(nodes, "llm", GoodLLM())
    result = risk_analyst_node(
        {
            "ticker": "NVDA",
            "forecast": _forecast(),
            "performance_trend": "BULLISH",
            "performance_repaired": False,
            "performance_analysis": "Trend: BULLISH\nAnalysis:\nUp path.",
            "news_sentiment": "MIXED",
            "news_summary": "Sentiment: MIXED\nAnalysis:\nMixed.",
            "financial_health": "ADEQUATE",
            "financial_analysis": "Health: ADEQUATE\nAnalysis:\nOk.",
            "financials": {"signals": {"leverage": "moderate"}},
        }
    )
    assert result["risk_level"] in {"CONTAINED", "MODERATE", "ELEVATED"}
    assert result["risk_guardrail_ok"] is True
    assert "Risk:" in result["risk_analysis"]
