"""Live news Research path — empty desk, tool → news agent → LLM synthesis."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agents.research_chat import answer_research_question
from src.agents.research_live_news import (
    answer_live_news_question,
    is_news_question,
    resolve_ticker_from_question,
)


class _ScriptedLLM:
    def __init__(self, texts: list[str] | str):
        self.texts = [texts] if isinstance(texts, str) else list(texts)
        self.calls = 0

    def invoke(self, messages):
        idx = min(self.calls, len(self.texts) - 1)
        self.calls += 1
        return AIMessage(content=self.texts[idx])


def _fake_news(ticker: str, *, limit: int = 6) -> dict:
    return {
        "status": "ok",
        "ticker": ticker.upper(),
        "provider": "test",
        "articles": [
            {
                "headline": f"{ticker.upper()} expands cloud partnership",
                "url": f"https://example.com/{ticker.lower()}-1",
                "published": "2026-10-01",
                "scope": "ticker",
            },
            {
                "headline": f"{ticker.upper()} shares react to earnings preview",
                "url": f"https://example.com/{ticker.lower()}-2",
                "published": "2026-10-02",
                "scope": "ticker",
            },
        ],
    }


def _fake_harness(*, ticker, news, news_raw, llm):
    return {
        "news_summary": (
            f"Sentiment: MIXED\nAnalysis:\n"
            f"Headlines for {ticker} show both partnership momentum and "
            f"earnings caution. Tone is MIXED on this live fetch."
        ),
        "news_sentiment": "MIXED",
        "news_guardrail_ok": True,
        "news_repaired": False,
    }


def test_resolve_ticker_brand_and_symbol():
    assert resolve_ticker_from_question("Latest headlines for Tesla") == "TSLA"
    assert resolve_ticker_from_question("Goldman Sachs news today") == "GS"
    assert resolve_ticker_from_question("what is the news on NVDA?") == "NVDA"
    assert is_news_question("Latest headlines for TSLA")
    assert not is_news_question("Explain disagreement")


def test_live_news_simple_headlines_skips_second_llm(monkeypatch):
    import src.agents.research_live_news as live

    monkeypatch.setattr(live, "get_news", _fake_news)
    monkeypatch.setattr(live, "run_news_harness", _fake_harness)

    llm = _ScriptedLLM("should not be called for simple headlines")
    result = answer_live_news_question(
        "Latest headlines for Tesla",
        llm=llm,
        news_fetcher=_fake_news,
        news_harness=_fake_harness,
    )
    assert result["path"] == "live_news"
    assert result["ticker"] == "TSLA"
    assert "get_news" in result["tools_used"]
    assert "news_agent" in result["tools_used"]
    assert "TSLA" in result["reply"]
    assert "MIXED" in result["reply"]
    assert "live" in result["reply"].lower()
    assert "cloud partnership" in result["reply"].lower()
    assert llm.calls == 0


def test_live_news_claim_question_uses_synthesis(monkeypatch):
    import src.agents.research_live_news as live

    monkeypatch.setattr(live, "get_news", _fake_news)
    monkeypatch.setattr(live, "run_news_harness", _fake_harness)

    llm = _ScriptedLLM(
        "### Live news · TSLA\n"
        "Source: live fetch (not a desk run packet).\n\n"
        "### Answer\n"
        "- No headline in this live fetch says market cap was wiped out.\n"
        "- Coverage mentions cloud partnership and earnings preview.\n\n"
        "NEXT: Explain disagreement|Walk through Risk"
    )
    result = answer_live_news_question(
        "Was Tesla’s market cap wiped out in half?",
        llm=llm,
        news_fetcher=_fake_news,
        news_harness=_fake_harness,
    )
    assert result["path"] == "live_news"
    assert llm.calls >= 1
    assert "wiped out" in result["reply"].lower() or "live" in result["reply"].lower()


def test_answer_research_routes_empty_desk_news(monkeypatch):
    import src.agents.research_live_news as live

    monkeypatch.setattr(live, "get_news", _fake_news)
    monkeypatch.setattr(live, "run_news_harness", _fake_harness)

    llm = _ScriptedLLM(
        "### Live news · GS\n"
        "Source: live fetch.\n\n"
        "### Summary\n"
        "- Sentiment MIXED from the Market expert note.\n\n"
        "NEXT: Why this confidence?|Forecast path"
    )
    result = answer_research_question(
        "What is the latest news on Goldman Sachs?",
        run_context=None,
        llm=llm,
    )
    assert result["mode"] == "research"
    assert "GS" in result["reply"] or "live" in result["reply"].lower()
    assert "get_news" in result["tools_used"]


def test_live_news_invented_url_falls_back():
    llm = _ScriptedLLM(
        "### News\n"
        "- See https://totally-fake.example/secret-scoop for the real story.\n\n"
        "NEXT: Explain disagreement|Walk through Risk"
    )
    result = answer_live_news_question(
        "Was Tesla market cap wiped out according to the news?",
        llm=llm,
        news_fetcher=_fake_news,
        news_harness=_fake_harness,
    )
    assert result["repaired"] is True
    assert result["ticker"] == "TSLA"
    assert "totally-fake.example" not in result["reply"]
    assert "TSLA" in result["reply"]
