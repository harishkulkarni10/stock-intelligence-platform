"""Unified desk assistant — routing, live-over-packet, guide vs desk."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agents.desk_assistant import answer_desk_assistant, is_product_guide_question
from src.agents.research_chat import answer_research_question
from src.agents.research_live_news import wants_live_news


class _ScriptedLLM:
    def __init__(self, text: str = "ok"):
        self.text = text

    def invoke(self, messages):
        return AIMessage(content=self.text)


def _fake_news(ticker: str, *, limit: int = 6) -> dict:
    return {
        "status": "ok",
        "ticker": ticker.upper(),
        "provider": "test",
        "articles": [
            {
                "headline": f"{ticker.upper()} beats earnings expectations",
                "url": f"https://example.com/{ticker.lower()}",
                "published": "2026-11-18",
                "scope": "ticker",
            }
        ],
    }


def _fake_harness(*, ticker, news, news_raw, llm):
    return {
        "news_summary": f"Sentiment: MIXED\nAnalysis:\n{ticker} earnings in headlines.",
        "news_sentiment": "MIXED",
        "news_repaired": False,
    }


def test_wants_live_news_claim_and_hot():
    assert wants_live_news("Was Tesla's market cap wiped out in half?")
    assert wants_live_news("What is the latest hot news of Nvidia from the past week")
    assert wants_live_news("Is NVIDIA being delisted based on latest headlines?")


def test_product_guide_vs_desk():
    assert is_product_guide_question("What does the Performance analyst do?")
    assert not is_product_guide_question("Explain disagreement")
    assert not is_product_guide_question("Latest headlines for TSLA")


def test_desk_packet_news_question_uses_live_not_packet_llm(monkeypatch):
    import src.agents.research_live_news as live

    monkeypatch.setattr(live, "get_news", _fake_news)
    monkeypatch.setattr(live, "run_news_harness", _fake_harness)

    packet = {
        "ticker": "NVDA",
        "recommendation": "NEUTRAL",
        "news_sentiment": "UNAVAILABLE",
        "news_summary": "Packet note says Jensen Huang fired — wrong.",
    }
    run_context = {
        "ticker": "NVDA",
        "desk_packets": {"NVDA": packet},
    }
    result = answer_research_question(
        "What is the latest hot news of Nvidia from the past week",
        run_context=run_context,
        llm=_ScriptedLLM(),
    )
    assert result.get("path") == "live_news"
    assert "get_news" in result.get("tools_used", [])
    assert "Jensen Huang fired" not in result["reply"]
    assert "Live news" in result["reply"]


def test_off_desk_market_cap_claim_routes_live(monkeypatch):
    import src.agents.research_live_news as live

    monkeypatch.setattr(live, "get_news", _fake_news)
    monkeypatch.setattr(live, "run_news_harness", _fake_harness)

    packet = {"ticker": "NVDA", "desk_packets": {"NVDA": {"ticker": "NVDA"}}}
    result = answer_research_question(
        "Was Tesla's market cap wiped out in half?",
        run_context=packet,
        llm=_ScriptedLLM(),
    )
    assert result.get("path") == "live_news"
    assert result.get("ticker") == "TSLA"


def test_bare_news_asks_for_ticker():
    result = answer_research_question(
        "news",
        run_context={"ticker": "NVDA", "desk_packets": {"NVDA": {"ticker": "NVDA"}}},
    )
    assert result.get("path") == "need_ticker"