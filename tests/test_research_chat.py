"""Unit tests for Research Co-Pilot (Phase 1)."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agents.research_chat import answer_research_question, resolve_chat_mode
from src.agents.research_guardrails import (
    ADVICE_META,
    OUT_OF_SCOPE,
    build_research_facts,
    deterministic_research_answer,
    is_advice_meta_question,
    is_advice_question,
    is_open_run_meta_question,
    prefers_deterministic_answer,
    strip_brochure_voice,
    strip_ungrounded_catalyst_sentences,
    validate_research_reply,
)
from src.agents.research_tools import (
    compare_labels,
    explain_confidence,
    normalize_desk_context,
    resolve_target_tickers,
    select_tools_for_question,
    truncate_run_context,
)
from src.monitoring.agent_eval import evaluate_research_fixtures


def _sample_packet(**overrides):
    packet = {
        "status": "ok",
        "ticker": "AAPL",
        "performance_trend": "SIDEWAYS",
        "performance_analysis": "Trend: SIDEWAYS\nAnalysis:\nFlat path.",
        "news_sentiment": "POSITIVE",
        "news_summary": "Sentiment: POSITIVE\nAnalysis:\nSupportive headlines.",
        "financial_health": "STRONG",
        "financial_analysis": "Health: STRONG\nAnalysis:\nSolid margins.",
        "risk_level": "MODERATE",
        "risk_analysis": "Risk: MODERATE\nAnalysis:\nSome uncertainty.",
        "final_report": "Stance: NEUTRAL\nConfidence: Medium\n\nExecutive summary:\nMixed tape.",
        "recommendation": "NEUTRAL",
        "confidence": "Medium",
        "predictions": {
            "last_close": 180.0,
            "model_source": "child",
            "horizon": 5,
            "predictions": [
                {"step": 1, "value": 180.5, "date": "2026-03-01"},
                {"step": 5, "value": 182.0, "date": "2026-03-05"},
            ],
        },
        "financials": {
            "metrics": {"revenue_growth": 0.08, "operating_margin": 0.3},
            "signals": {"liquidity": "ok"},
        },
        "risk": {"metrics": {"realized_vol": 0.22}},
        "news": {
            "articles": [
                {
                    "title": "Apple updates product line",
                    "url": "https://example.com/aapl-1",
                    "published": "2026-03-01",
                }
            ]
        },
        "company": {"name": "Apple Inc.", "sector": "Technology"},
        "cached": False,
    }
    packet.update(overrides)
    return packet


class _ScriptedLLM:
    def __init__(self, texts: list[str] | str):
        self.texts = [texts] if isinstance(texts, str) else list(texts)
        self.calls = 0

    def invoke(self, messages):
        idx = min(self.calls, len(self.texts) - 1)
        self.calls += 1
        return AIMessage(content=self.texts[idx])


def test_truncate_requires_ticker():
    assert truncate_run_context({}) is None
    assert truncate_run_context(_sample_packet())["ticker"] == "AAPL"


def test_compare_labels_flags_disagreement():
    result = compare_labels(_sample_packet())
    assert result["agreement"] == "disagreement"
    labels = {row["dimension"]: row["label"] for row in result["rows"]}
    assert labels["Performance trend"] == "SIDEWAYS"
    assert labels["News sentiment"] == "POSITIVE"


def test_advice_question_detected():
    assert is_advice_question("Should I buy AAPL?")
    assert not is_advice_question("Why is stance NEUTRAL if news is POSITIVE?")


def test_no_run_returns_prompt():
    result = answer_research_question("Explain disagreement", run_context=None)
    assert result["mode"] == "research"
    assert "No open run" in result["reply"]
    assert result["out_of_scope"] is False


def test_buy_question_out_of_scope():
    result = answer_research_question(
        "Should I buy AAPL?",
        run_context=_sample_packet(),
        llm=_ScriptedLLM("should not be called"),
    )
    assert result["out_of_scope"] is True
    assert result["reply"] == OUT_OF_SCOPE


def test_invented_price_fails_validation():
    facts = build_research_facts(_sample_packet())
    bad = (
        "### Price\n"
        "The stock will hit 999.99 tomorrow based on secret data.\n"
    )
    check = validate_research_reply(bad, facts)
    assert check.ok is False
    assert any(err.startswith("unknown_number") for err in check.errors)


def test_harness_repairs_invented_price():
    llm = _ScriptedLLM(
        "### Outlook\n"
        "- Price target 999.99 is certain.\n"
        "- News is POSITIVE while stance is NEUTRAL.\n"
    )
    result = answer_research_question(
        "Why is stance NEUTRAL if news is POSITIVE?",
        run_context=_sample_packet(),
        llm=llm,
    )
    assert result["out_of_scope"] is False
    assert "999.99" not in result["reply"]
    assert result["repaired"] is True or "NEUTRAL" in result["reply"]


def test_disagreement_answer_mentions_both_labels():
    llm = _ScriptedLLM(
        "### Disagreement\n"
        "- Performance trend is SIDEWAYS so stance is NEUTRAL.\n"
        "- News sentiment is POSITIVE on this run.\n"
        "- That tension is useful research signal.\n\n"
        "NEXT: Walk through Risk|Forecast path"
    )
    result = answer_research_question(
        "Explain disagreement across specialists",
        run_context=_sample_packet(),
        llm=llm,
    )
    reply = result["reply"]
    assert "SIDEWAYS" in reply or "NEUTRAL" in reply
    assert "POSITIVE" in reply
    assert result["out_of_scope"] is False
    assert any("compare_labels" in t for t in result["tools_used"])


def test_deterministic_fallback_grounded():
    reply = deterministic_research_answer(
        _sample_packet(), question="Explain disagreement"
    )
    facts = build_research_facts(_sample_packet())
    assert validate_research_reply(reply, facts).ok is True
    assert "POSITIVE" in reply
    assert "NEUTRAL" in reply or "SIDEWAYS" in reply


def test_resolve_chat_mode():
    packet = _sample_packet()
    assert resolve_chat_mode("guide", packet) == "guide"
    assert resolve_chat_mode("research", None) == "research"
    assert resolve_chat_mode("auto", packet) == "research"
    assert resolve_chat_mode("auto", None) == "guide"


def test_select_tools_for_risk_question():
    plan = select_tools_for_question("Walk through Risk on this run")
    names = [name for name, _ in plan]
    assert "get_risk_metrics" in names
    assert "get_run_overview" in names


def test_help_chat_routes_research(monkeypatch):
    from fastapi.testclient import TestClient

    from backend.main import app
    from src.agents import research_chat as research_mod

    monkeypatch.setattr(
        research_mod,
        "answer_research_question",
        lambda message, history=None, run_context=None, llm=None: {
            "reply": "### Disagreement\n- NEUTRAL vs POSITIVE",
            "out_of_scope": False,
            "mode": "research",
            "repaired": False,
            "tools_used": ["compare_labels"],
        },
    )
    client = TestClient(app)
    response = client.post(
        "/help-chat",
        json={
            "message": "Explain disagreement",
            "mode": "research",
            "analyze_context": _sample_packet(),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "research"
    assert "POSITIVE" in body["reply"] or "NEUTRAL" in body["reply"]


def test_evaluate_research_fixtures():
    result = evaluate_research_fixtures()
    assert result["ok"] is True, result


def test_explain_confidence_not_only_mixed_news():
    packet = _sample_packet(
        performance_trend="BULLISH",
        recommendation="BULLISH",
        news_sentiment="MIXED",
        confidence="Medium",
        predictions={
            "last_close": 100.0,
            "model_source": "parent",
            "horizon": 5,
            "predictions": [
                {"step": 1, "value": 100.5},
                {"step": 5, "value": 101.0},
            ],
        },
    )
    meta = explain_confidence(packet)
    assert meta["confidence"] == "Medium"
    assert meta["model_source"] == "parent"
    assert "MIXED news by itself" in meta["meaning"] or "not" in meta["meaning"].lower()


def test_confidence_question_uses_deterministic_path():
    llm = _ScriptedLLM("I am Research Co-Pilot. Medium only because news is MIXED.")
    result = answer_research_question(
        "Why this confidence?",
        run_context=_sample_packet(confidence="Medium"),
        llm=llm,
    )
    assert llm.calls == 0
    reply = result["reply"]
    assert "I am Research Co-Pilot" not in reply
    assert "Medium" in reply
    assert "persistence" in reply.lower() or "model" in reply.lower()
    assert "NEXT:" in reply
    assert prefers_deterministic_answer("What does it mean when confidence is medium")


def test_strip_brochure_voice():
    dirty = (
        "I am Research Co-Pilot for Stock Intelligence. I answer questions regarding the run.\n\n"
        "### Labels\n"
        "- Trend BULLISH\n"
    )
    clean = strip_brochure_voice(dirty)
    assert "I am Research Co-Pilot" not in clean
    assert "Trend BULLISH" in clean


def test_glossary_question_covers_projected_move():
    reply = deterministic_research_answer(
        _sample_packet(),
        question="What does it mean when trend is bullish and projected move is positive?",
    )
    assert "Projected move" in reply or "projected move" in reply.lower()
    assert "SIDEWAYS" in reply or "Trend" in reply
    assert "NEXT:" in reply


def test_open_run_meta_question_scripted():
    assert is_open_run_meta_question("why am I talking to you")
    assert is_open_run_meta_question("now tell me why I am here")
    llm = _ScriptedLLM("### Analysis Scope\nThis session is dedicated to AAPL.")
    result = answer_research_question(
        "how do you know I am here for AMZN analysis",
        run_context=_sample_packet(ticker="AMZN", recommendation="BULLISH"),
        llm=llm,
    )
    assert llm.calls == 0
    reply = result["reply"]
    assert "AMZN" in reply
    assert "Analysis Scope" not in reply
    assert "desk" in reply.lower() or "scope" in reply.lower()
    assert "NEXT:" in reply


def test_desk_packets_answer_named_ticker_without_opening():
    """Active view BLK; ask about ORCL → ORCL packet facts, no click-to-open tip."""
    blk = _sample_packet(
        ticker="BLK",
        performance_trend="SIDEWAYS",
        recommendation="NEUTRAL",
        confidence="Medium",
        company={"name": "BlackRock, Inc."},
    )
    amzn = _sample_packet(
        ticker="AMZN",
        performance_trend="BULLISH",
        recommendation="BULLISH",
        confidence="Medium",
        news_sentiment="MIXED",
        company={"name": "Amazon.com, Inc."},
        predictions={
            "last_close": 200.0,
            "model_source": "parent",
            "horizon": 5,
            "predictions": [
                {"step": 1, "value": 201.0},
                {"step": 5, "value": 205.0},
            ],
        },
    )
    orcl = _sample_packet(
        ticker="ORCL",
        performance_trend="BEARISH",
        recommendation="BEARISH",
        confidence="Low",
        news_sentiment="NEGATIVE",
        company={"name": "Oracle Corporation"},
        predictions={
            "last_close": 150.0,
            "model_source": "persistence",
            "horizon": 5,
            "predictions": [
                {"step": 1, "value": 149.0},
                {"step": 5, "value": 145.0},
            ],
        },
        performance_repaired=True,
    )
    desk = {
        "active_ticker": "BLK",
        "desk_tickers": ["BLK", "AMZN", "ORCL"],
        "desk_packets": {"BLK": blk, "AMZN": amzn, "ORCL": orcl},
    }
    ctx = normalize_desk_context(desk)
    assert ctx is not None
    assert resolve_target_tickers("Why was ORCL confidence Low?", ctx) == ["ORCL"]
    assert resolve_target_tickers("tell me about Amazon confidence", ctx) == ["AMZN"]
    assert resolve_target_tickers("Why this confidence?", ctx) == ["BLK"]

    llm = _ScriptedLLM("should not be called for deterministic confidence")
    result = answer_research_question(
        "Why was ORCL confidence Low?",
        run_context=desk,
        llm=llm,
    )
    assert llm.calls == 0
    reply = result["reply"]
    assert "ORCL" in reply or "Low" in reply
    assert "click" not in reply.lower()
    assert "persistence" in reply.lower() or "Low" in reply
    assert "NEXT:" in reply

    # Company-name routing
    result2 = answer_research_question(
        "What does it mean when Oracle trend is bearish?",
        run_context=desk,
        llm=llm,
    )
    assert "ORCL" in result2["reply"] or "BEARISH" in result2["reply"]


def test_missing_ticker_not_on_desk():
    desk = {
        "active_ticker": "BLK",
        "desk_packets": {
            "BLK": _sample_packet(ticker="BLK", recommendation="NEUTRAL"),
        },
    }
    result = answer_research_question(
        "AMZN is also in the session - check it properly",
        run_context=desk,
        llm=_ScriptedLLM("nope"),
    )
    assert "Not on this desk" in result["reply"] or "not among" in result["reply"].lower()
    assert "AMZN" in result["reply"]


def test_multi_compare_projected_moves():
    desk = {
        "active_ticker": "JPM",
        "desk_packets": {
            "JPM": _sample_packet(
                ticker="JPM",
                performance_trend="BULLISH",
                recommendation="BULLISH",
                company={"name": "JPMorgan Chase & Co."},
                predictions={
                    "last_close": 200.0,
                    "model_source": "parent",
                    "horizon": 5,
                    "predictions": [
                        {"step": 1, "value": 201.0},
                        {"step": 5, "value": 201.6},
                    ],
                },
            ),
            "KO": _sample_packet(
                ticker="KO",
                performance_trend="BULLISH",
                recommendation="BULLISH",
                company={"name": "The Coca-Cola Company"},
                predictions={
                    "last_close": 90.0,
                    "model_source": "parent",
                    "horizon": 5,
                    "predictions": [
                        {"step": 1, "value": 90.5},
                        {"step": 5, "value": 90.7},
                    ],
                },
            ),
            "WMT": _sample_packet(
                ticker="WMT",
                performance_trend="BULLISH",
                recommendation="BULLISH",
                company={"name": "Walmart Inc."},
                predictions={
                    "last_close": 100.0,
                    "model_source": "parent",
                    "horizon": 5,
                    "predictions": [
                        {"step": 1, "value": 100.2},
                        {"step": 5, "value": 100.5},
                    ],
                },
            ),
            "PLTR": _sample_packet(
                ticker="PLTR",
                performance_trend="BULLISH",
                recommendation="BULLISH",
                company={"name": "Palantir Technologies Inc."},
                predictions={
                    "last_close": 50.0,
                    "model_source": "parent",
                    "horizon": 5,
                    "predictions": [
                        {"step": 1, "value": 50.2},
                        {"step": 5, "value": 50.5},
                    ],
                },
            ),
        },
    }
    result = answer_research_question(
        "Compare KO, JPM, Walmart and palantir projected move, what they mean",
        run_context=desk,
        llm=_ScriptedLLM("should not run"),
    )
    assert "Not on this desk" not in result["reply"] or "AMD" in result["reply"]
    reply = result["reply"]
    assert "KO:" in reply and "JPM:" in reply
    assert "WMT:" in reply and "PLTR:" in reply
    assert "How to read projected move" in reply
    assert "should not run" not in reply


def test_multi_risk_notes_missing_ticker():
    desk = {
        "active_ticker": "JPM",
        "desk_packets": {
            "JPM": _sample_packet(
                ticker="JPM",
                risk_level="CONTAINED",
                recommendation="BULLISH",
                risk={"metrics": {"realized_vol": 0.15}},
            ),
            "WMT": _sample_packet(
                ticker="WMT",
                risk_level="MODERATE",
                recommendation="BULLISH",
                company={"name": "Walmart Inc."},
                risk={"metrics": {"realized_vol": 0.22}},
            ),
        },
    }
    result = answer_research_question(
        "walk through the risk on walmart, AMD and JPM",
        run_context=desk,
        llm=_ScriptedLLM("nope"),
    )
    reply = result["reply"]
    assert "WMT" in reply and "JPM" in reply
    assert "AMD" in reply
    assert "no finished packet" in reply.lower() or "Not on this desk" in reply
    # Must not invent AMD volatility figures
    assert "50.9" not in reply


def test_mixed_news_focused_glossary():
    result = answer_research_question(
        "What does it mean when news is MIXED?",
        run_context=_sample_packet(ticker="MU", news_sentiment="MIXED"),
        llm=_ScriptedLLM("nope"),
    )
    reply = result["reply"]
    assert "MIXED" in reply
    assert "Labels on this run" not in reply
    assert "headline tone" in reply.lower() or "News" in reply


def test_advice_meta_not_blunt_oos():
    assert is_advice_meta_question("why don't you give sell advice")
    result = answer_research_question(
        "why don't you give sell advice",
        run_context=_sample_packet(),
        llm=_ScriptedLLM("nope"),
    )
    assert result["out_of_scope"] is True
    assert result["reply"] == ADVICE_META
    assert "not a broker" in result["reply"].lower()


def test_why_bullish_deterministic_no_kfc():
    packet = _sample_packet(
        ticker="KO",
        performance_trend="BULLISH",
        recommendation="BULLISH",
        performance_analysis="Trend: BULLISH\nStair-step higher floors.",
    )
    result = answer_research_question(
        "Why is KO bullish?",
        run_context=packet,
        llm=_ScriptedLLM(
            "KO is bullish because of a major upcoming partnership with KFC in 2027."
        ),
    )
    assert "KFC" not in result["reply"]
    assert "2027" not in result["reply"]
    assert "BULLISH" in result["reply"]


def test_strip_ungrounded_catalyst():
    packet = _sample_packet(performance_analysis="Trend: BULLISH only.")
    dirty = "Long-term sentiment is bolstered by a major upcoming partnership with KFC."
    clean = strip_ungrounded_catalyst_sentences(dirty, packet)
    assert "KFC" not in clean


def test_coca_cola_brand_resolves_to_ko():
    desk = {
        "active_ticker": "JPM",
        "desk_packets": {
            "JPM": _sample_packet(
                ticker="JPM",
                recommendation="BULLISH",
                company={"name": "JPMorgan Chase & Co."},
            ),
            "KO": _sample_packet(
                ticker="KO",
                performance_trend="BULLISH",
                recommendation="BULLISH",
                confidence="Medium",
                company={"name": "The Coca-Cola Company"},
                predictions={
                    "last_close": 93.0,
                    "model_source": "parent",
                    "horizon": 5,
                    "predictions": [
                        {"step": 1, "value": 93.2},
                        {"step": 5, "value": 93.8},
                    ],
                },
            ),
        },
    }
    ctx = normalize_desk_context(desk)
    assert ctx is not None
    assert resolve_target_tickers("tell me about CoCa Cola trend", ctx) == ["KO"]
    assert resolve_target_tickers("what about coke risk?", ctx) == ["KO"]

    result = answer_research_question(
        "tell me about CoCa Cola trend",
        run_context=desk,
        llm=_ScriptedLLM(
            "### KO trend\n"
            "- Performance trend is BULLISH.\n"
            "- Stance is BULLISH.\n\n"
            "NEXT: Explain disagreement|Walk through Risk"
        ),
    )
    assert "Not on this desk" not in result["reply"]
    assert "COCA" not in result["reply"]
    assert "KO" in result["reply"] or "BULLISH" in result["reply"]


def test_truncate_keeps_desk_tickers():
    packet = truncate_run_context(
        _sample_packet(ticker="BLK", desk_tickers=["BLK", "AMZN"])
    )
    assert packet is not None
    assert packet["ticker"] == "BLK"
    assert packet["desk_tickers"] == ["BLK", "AMZN"]
