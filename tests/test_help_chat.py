from langchain_core.messages import AIMessage

from src.agents.help_chat import (
    OUT_OF_SCOPE,
    SAFE_OVERVIEW,
    answer_help_question,
    sanitize_help_reply,
)


class _ScriptedLLM:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.text)


def test_help_chat_answers_from_knowledge():
    llm = _ScriptedLLM(
        "### Financial analyst\n"
        "Reviews fundamentals and writes health, strengths, and weaknesses.\n\n"
        "### Focus\n"
        "- Revenue and margins\n"
        "- Cash and leverage\n"
    )
    result = answer_help_question("What does the Financial agent do?", llm=llm)
    assert result["out_of_scope"] is False
    assert "Financial" in result["reply"]
    assert "###" in result["reply"]
    assert llm.calls == 1


def test_help_chat_out_of_scope():
    llm = _ScriptedLLM(OUT_OF_SCOPE)
    result = answer_help_question("Should I buy NVDA tomorrow?", llm=llm)
    assert result["out_of_scope"] is True
    assert result["reply"] == OUT_OF_SCOPE
    assert "internal" not in result["reply"].lower()


def test_sanitize_strips_bold_stars():
    dirty = (
        "**Agent roles:** Ask how the analysts work.\n"
        "**Feature navigation:** Ask about On your desk.\n"
    )
    reply, out = sanitize_help_reply(dirty, question="what can I ask you")
    assert out is False
    assert "**" not in reply
    assert "Agent roles" in reply

    dirty = (
        "**How the system works:**\n"
        "- Forecast uses LSTM models\n"
        "- Cache is Redis prefix analyze-perf-news-fin-v1\n"
    )
    reply, out = sanitize_help_reply(dirty, question="tell me about the application")
    assert out is False
    assert reply == SAFE_OVERVIEW
    assert "Redis" not in reply
    assert "LSTM" not in reply


def test_help_chat_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    from backend.main import app
    from src.agents import help_chat as help_mod

    monkeypatch.setattr(
        help_mod,
        "answer_help_question",
        lambda message, history=None, llm=None: {
            "reply": "### Performance analyst\nExplains the forecast path.",
            "out_of_scope": False,
        },
    )
    client = TestClient(app)
    response = client.post("/help-chat", json={"message": "What is agent 1?"})
    assert response.status_code == 200
    body = response.json()
    assert body["out_of_scope"] is False
    assert "Performance" in body["reply"]
