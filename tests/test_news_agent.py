"""Unit tests for Market Expert (agent 2) guardrails and harness."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agents.news_guardrails import (
    build_news_facts,
    run_news_harness,
    validate_news_analysis,
)
from src.agents.nodes import market_expert_node
from src.agents.tools import format_news_for_prompt
from src.monitoring.agent_eval import evaluate_news_fixtures

_LONG_ANALYSIS = (
    "Coverage around NVIDIA is constructive and internally consistent. The earnings "
    "beat tied to GPU demand points to customers still expanding AI infrastructure, "
    "and a careful reader should still open the primary article to confirm how broad "
    "the beat was. That does not remove cyclical or competitive risk, but for a "
    "short-horizon news read the tape is supportive rather than conflicted. Taken "
    "together, the available headline justifies a POSITIVE sentiment label while "
    "leaving room for macro or peer surprises that are not spelled out in this sample."
)


def _ok_news() -> dict:
    return {
        "status": "ok",
        "ticker": "NVDA",
        "provider": "fixture",
        "articles": [
            {
                "date": "2026-09-01",
                "headline": "NVIDIA beats earnings estimates on strong GPU demand",
                "summary": "Data center revenue surged as cloud customers expanded AI clusters.",
                "url": "https://example.com/a",
            }
        ],
    }


def test_unavailable_news_forces_unavailable_sentiment():
    facts = build_news_facts(
        {"status": "error", "ticker": "NVDA", "articles": [], "error": "down"},
        ticker="NVDA",
    )
    assert facts.allowed_sentiment == "UNAVAILABLE"
    bad = "Sentiment: POSITIVE\nDrivers: - secret boom\nCaveat: none"
    assert validate_news_analysis(bad, facts).ok is False


def test_grounded_positive_summary_passes():
    news = _ok_news()
    facts = build_news_facts(news)
    text = (
        "Sentiment: POSITIVE\n"
        "Headlines:\n"
        "- NVIDIA beats earnings estimates on strong GPU demand\n"
        f"Analysis:\n{_LONG_ANALYSIS}\n"
        "Implications:\n"
        "- Demand tone currently leans supportive for NVIDIA.\n"
        "Caveats: Single-source snapshot."
    )
    assert validate_news_analysis(text, facts).ok is True


def test_invented_headline_fails():
    news = _ok_news()
    facts = build_news_facts(news)
    text = (
        "Sentiment: POSITIVE\n"
        "Headlines:\n"
        "- Apple acquires a secret quantum chip startup in Zurich for forty billion dollars overnight\n"
        f"Analysis:\n{_LONG_ANALYSIS}\n"
        "Caveats: none"
    )
    result = validate_news_analysis(text, facts)
    assert result.ok is False


def test_short_drivers_only_fails():
    news = _ok_news()
    facts = build_news_facts(news)
    text = (
        "Sentiment: POSITIVE\n"
        "Drivers: - NVIDIA beats earnings estimates on strong GPU demand\n"
        "Caveat: Limited article set."
    )
    assert validate_news_analysis(text, facts).ok is False


def test_harness_recovers_when_news_missing():
    news = {"status": "error", "ticker": "MSFT", "articles": [], "error": "empty"}

    class BadLLM:
        def invoke(self, messages):
            return AIMessage(content="Sentiment: POSITIVE\nDrivers: - fake rally\nCaveat: x")

    out = run_news_harness(
        ticker="MSFT",
        news=news,
        news_raw=format_news_for_prompt(news),
        llm=BadLLM(),
        max_attempts=1,
    )
    assert out["news_sentiment"] == "UNAVAILABLE"
    assert out["news_guardrail_ok"] is True
    assert out["news_summary"].startswith("Sentiment: UNAVAILABLE")


def test_market_expert_node_uses_harness(monkeypatch):
    from src.agents import nodes

    news = _ok_news()

    class GoodLLM:
        def invoke(self, messages):
            return AIMessage(
                content=(
                    "Sentiment: POSITIVE\n"
                    "Headlines:\n"
                    "- NVIDIA beats earnings estimates on strong GPU demand\n"
                    f"Analysis:\n{_LONG_ANALYSIS}\n"
                    "Implications:\n"
                    "- Demand tone currently leans supportive for NVIDIA.\n"
                    "Caveats: Limited article set."
                )
            )

    monkeypatch.setattr(nodes, "llm", GoodLLM())
    monkeypatch.setattr(nodes, "get_news", lambda ticker: news)
    result = market_expert_node({"ticker": "NVDA"})
    assert result["news_sentiment"] == "POSITIVE"
    assert result["news_guardrail_ok"] is True
    assert "Analysis:" in result["news_summary"]


def test_news_eval_fixtures_pass():
    report = evaluate_news_fixtures()
    assert report["ok"] is True, report
