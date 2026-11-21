"""Live news Research path — empty desk, tool → news agent → LLM synthesis."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agents.research_chat import answer_research_question
from src.agents.research_live_news import (
    answer_live_news_question,
    is_news_question,
    resolve_ticker_from_context,
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


def _fake_news(ticker: str, *, limit: int = 6, **_: object) -> dict:
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


def test_resolve_ticker_brand_and_symbol():
    assert resolve_ticker_from_question("Latest headlines for Tesla") == "TSLA"
    assert resolve_ticker_from_question("Goldman Sachs news today") == "GS"
    assert resolve_ticker_from_question("what is the news on NVDA?") == "NVDA"
    assert is_news_question("Latest headlines for TSLA")
    assert not is_news_question("Explain disagreement")


def test_live_news_simple_headlines_zero_llm_fast_path(monkeypatch):
    import src.agents.research_live_news as live

    monkeypatch.setattr(live, "get_news", _fake_news)

    llm = _ScriptedLLM("unused — fast path skips chat synthesis")
    result = answer_live_news_question(
        "Latest headlines for Tesla",
        llm=llm,
        news_fetcher=_fake_news,
    )
    assert result["path"] == "live_news"
    assert result["ticker"] == "TSLA"
    assert result.get("fast_path") == "zero_llm"
    assert result.get("llm_calls") == 0
    assert "get_news" in result["tools_used"]
    assert "news_agent" not in result["tools_used"]
    assert "**TSLA**" in result["reply"]
    assert "worth a look" in result["reply"].lower()
    assert "cloud partnership" in result["reply"].lower()
    assert llm.calls == 0


def test_give_me_headlines_for_brand_zero_llm_with_history(monkeypatch):
    import src.agents.research_live_news as live

    monkeypatch.setattr(live, "get_news", _fake_news)
    history = [
        {"role": "user", "content": "Latest headlines for NVDA"},
        {"role": "assistant", "content": "**NVDA** — sample\n\n**Worth a look**\n- x"},
    ]
    llm = _ScriptedLLM("must not run")
    result = answer_live_news_question(
        "give me headlines for Netflix",
        history=history,
        llm=llm,
        news_fetcher=_fake_news,
    )
    assert result.get("fast_path") == "zero_llm"
    assert result.get("llm_calls") == 0
    assert "worth a look" in result["reply"].lower()
    assert llm.calls == 0


def test_live_news_follow_up_bad_zero_llm(monkeypatch):
    import src.agents.research_live_news as live

    monkeypatch.setattr(live, "get_news", _fake_news)
    history = [
        {"role": "user", "content": "what's happening with Nvidia"},
        {"role": "assistant", "content": "**NVDA** — themes\n\n**Worth a look**\n- a"},
    ]
    llm = _ScriptedLLM("must not run")
    result = answer_live_news_question(
        "what's bad?",
        history=history,
        llm=llm,
        news_fetcher=_fake_news,
    )
    assert result.get("fast_path") == "zero_llm"
    assert llm.calls == 0


def test_live_news_claim_question_uses_synthesis(monkeypatch):
    import src.agents.research_live_news as live

    monkeypatch.setattr(live, "get_news", _fake_news)
    llm = _ScriptedLLM("must not run — headlines do not support claim")
    result = answer_live_news_question(
        "Was Tesla’s market cap wiped out in half?",
        llm=llm,
        news_fetcher=_fake_news,
    )
    assert result["path"] == "live_news"
    assert llm.calls == 0
    assert result.get("llm_calls") == 0
    assert result.get("fast_path") == "claim_zero_llm"
    assert "**no**" in result["reply"].lower()
    assert "market cap" in result["reply"].lower()


def test_live_news_bankruptcy_claim_zero_llm(monkeypatch):
    import src.agents.research_live_news as live

    monkeypatch.setattr(live, "get_news", _fake_news)
    llm = _ScriptedLLM("unused")
    result = answer_live_news_question(
        "Did Goldman Sachs declare bankruptcy this morning according to the news?",
        llm=llm,
        news_fetcher=lambda t, **_: _fake_news("GS"),
    )
    assert result.get("fast_path") == "claim_zero_llm"
    assert "**no**" in result["reply"].lower()
    assert "bankruptcy" in result["reply"].lower()
    assert llm.calls == 0


def test_delisted_in_question_uses_claim_zero_llm(monkeypatch):
    import src.agents.research_live_news as live

    monkeypatch.setattr(live, "get_news", _fake_news)
    llm = _ScriptedLLM("unused")
    result = answer_live_news_question(
        "Is NVIDIA being delisted based on latest headlines?",
        llm=llm,
        news_fetcher=lambda t, **_: _fake_news("NVDA"),
    )
    assert result.get("fast_path") == "claim_zero_llm"
    assert "**no**" in result["reply"].lower()
    assert "delist" in result["reply"].lower()
    assert llm.calls == 0


def test_goldman_headlines_use_standard_opening_not_topic_filter(monkeypatch):
    import src.agents.research_live_news as live

    monkeypatch.setattr(live, "get_news", _fake_news)
    llm = _ScriptedLLM("unused")
    result = answer_live_news_question(
        "What is the latest news on Goldman Sachs?",
        llm=llm,
        news_fetcher=lambda t, **_: _fake_news("GS"),
    )
    assert "headlines center on" in result["reply"].lower()
    assert "matches goldman" not in result["reply"].lower()
    assert llm.calls == 0


def test_ipo_token_not_ticker():
    assert resolve_ticker_from_question("Did Anthropic's IPO launch today?") is None


def test_anthropic_ipo_uses_session_ticker(monkeypatch):
    import src.agents.research_live_news as live

    def _nvda_anthropic_news(ticker: str, **_: object) -> dict:
        return {
            "status": "ok",
            "ticker": "NVDA",
            "provider": "test",
            "articles": [
                {
                    "headline": "Cathie Wood buys Anthropic stake while trimming NVDA",
                    "url": "https://example.com/anthropic",
                    "scope": "ticker",
                },
            ],
        }

    monkeypatch.setattr(live, "get_news", _nvda_anthropic_news)
    history = [
        {"role": "user", "content": "Headlines related to Nvidia"},
        {
            "role": "assistant",
            "content": "**NVDA** — themes\n\n**Worth a look**\n- Cathie Wood buys Anthropic",
        },
        {"role": "user", "content": "headlines on Tesla"},
        {"role": "assistant", "content": "**TSLA** — themes\n\n**Worth a look**\n- Cybercab"},
    ]
    assert (
        resolve_ticker_from_context(
            "tell me news about Anthropic IPO",
            history,
        )
        == "NVDA"
    )
    llm = _ScriptedLLM("unused")
    result = answer_live_news_question(
        "tell me news about Anthropic IPO",
        history=history,
        llm=llm,
        news_fetcher=_nvda_anthropic_news,
    )
    assert result["ticker"] == "NVDA"
    assert result.get("fast_path") == "private_co_zero_llm"
    assert "no ipo" in result["reply"].lower()
    assert "anthropic" in result["reply"].lower()
    assert "david ellison" not in result["reply"].lower()


def test_live_news_topic_follow_up_filters_cybercab(monkeypatch):
    import src.agents.research_live_news as live

    def _tsla_news(ticker: str, **_: object) -> dict:
        return {
            "status": "ok",
            "ticker": "TSLA",
            "provider": "test",
            "articles": [
                {
                    "headline": "Tesla Cybercab arrives in Dallas amid robo-taxi push",
                    "url": "https://example.com/cybercab",
                    "scope": "ticker",
                },
                {
                    "headline": "Musk updates Optimus robot dishwasher timeline",
                    "url": "https://example.com/optimus",
                    "scope": "ticker",
                },
                {
                    "headline": "Michael Burry adds BYD at five-year lows",
                    "url": "https://example.com/byd",
                    "scope": "ticker",
                },
            ],
        }

    monkeypatch.setattr(live, "get_news", _tsla_news)
    history = [
        {"role": "user", "content": "Latest headlines for TSLA"},
        {
            "role": "assistant",
            "content": "**TSLA** — sample\n\n**Worth a look**\n- Tesla Cybercab arrives",
        },
    ]
    llm = _ScriptedLLM("unused")
    result = answer_live_news_question(
        "What about the Cybercab stories?",
        history=history,
        llm=llm,
        news_fetcher=_tsla_news,
    )
    assert result.get("fast_path") == "zero_llm"
    assert llm.calls == 0
    reply_low = result["reply"].lower()
    assert "cybercab" in reply_low
    assert "burry" not in reply_low


def test_answer_research_routes_empty_desk_news(monkeypatch):
    import src.agents.research_live_news as live

    monkeypatch.setattr(live, "get_news", _fake_news)
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


def test_live_news_invented_url_falls_back(monkeypatch):
    import src.agents.research_live_news as live

    monkeypatch.setattr(live, "get_news", _fake_news)

    def _affirming_news(ticker: str, **_: object) -> dict:
        data = _fake_news(ticker)
        data["articles"] = [
            {
                "headline": "Analysts debate whether TSLA market cap was halved overnight",
                "url": "https://example.com/halved",
                "scope": "ticker",
            }
        ]
        return data

    llm = _ScriptedLLM(
        "### News\n"
        "- See https://totally-fake.example/secret-scoop for the real story.\n\n"
        "NEXT: Explain disagreement|Walk through Risk"
    )
    result = answer_live_news_question(
        "Was Tesla market cap wiped out according to the news?",
        llm=llm,
        news_fetcher=_affirming_news,
    )
    assert result["repaired"] is True
    assert result["ticker"] == "TSLA"
    assert "totally-fake.example" not in result["reply"]
    assert "TSLA" in result["reply"]
    assert "**no**" in result["reply"].lower()
