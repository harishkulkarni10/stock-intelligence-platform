"""Phase 0 — /help-chat observability (meta.wall_ms, path, ticker)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def test_help_chat_research_includes_meta_wall_ms(monkeypatch):
    import src.agents.desk_assistant as assistant_mod

    monkeypatch.setattr(
        assistant_mod,
        "answer_desk_assistant",
        lambda message, history=None, run_context=None, llm=None: {
            "reply": "### Baseline\n- ok",
            "out_of_scope": False,
            "mode": "assistant",
            "repaired": False,
            "tools_used": ["get_news"],
            "path": "live_news",
            "ticker": "TSLA",
            "fast_path": "harness_only",
        },
    )
    client = TestClient(app)
    response = client.post(
        "/help-chat",
        json={"message": "Latest headlines for TSLA", "mode": "research"},
    )
    assert response.status_code == 200
    body = response.json()
    meta = body.get("meta") or {}
    assert isinstance(meta.get("wall_ms"), (int, float))
    assert meta["wall_ms"] >= 0
    assert meta.get("path") == "live_news"
    assert meta.get("ticker") == "TSLA"
    assert meta.get("fast_path") == "harness_only"


def test_help_chat_guide_meta_wall_ms_only(monkeypatch):
    import src.agents.desk_assistant as assistant_mod

    monkeypatch.setattr(
        assistant_mod,
        "answer_desk_assistant",
        lambda message, history=None, run_context=None, llm=None: {
            "reply": "### Guide\n- ok",
            "out_of_scope": False,
            "mode": "assistant",
            "tools_used": [],
            "repaired": False,
            "path": "product_guide",
        },
    )
    client = TestClient(app)
    response = client.post(
        "/help-chat", json={"message": "What does the Performance analyst do?"}
    )
    body = response.json()
    meta = body.get("meta") or {}
    assert "wall_ms" in meta
    assert meta.get("path") in (None, "product_guide")