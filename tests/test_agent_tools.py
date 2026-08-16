from __future__ import annotations

from src.agents import tools


def test_get_forecast_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "parent_artifact_dir", lambda cfg=None: tmp_path / "parent")
    monkeypatch.setattr(
        tools, "child_artifact_dir", lambda ticker, cfg=None: tmp_path / ticker
    )

    result = tools.get_forecast("NVDA")

    assert result["status"] == "missing_model"
    assert result["ticker"] == "NVDA"


def test_get_forecast_prefers_child(monkeypatch, tmp_path):
    parent = tmp_path / "parent"
    child = tmp_path / "NVDA"
    parent.mkdir()
    child.mkdir()
    (parent / "model.pt").write_bytes(b"parent")
    (child / "model.pt").write_bytes(b"child")

    monkeypatch.setattr(tools, "parent_artifact_dir", lambda cfg=None: parent)
    monkeypatch.setattr(tools, "child_artifact_dir", lambda ticker, cfg=None: child)
    monkeypatch.setattr(
        tools,
        "predict_child",
        lambda ticker, horizon: {
            "ticker": ticker,
            "horizon": horizon,
            "model_version": "child-1",
            "model_type": "child",
            "last_close": 100.0,
            "last_date": "2026-08-14",
            "history": [],
            "predictions": [
                {"step": 1, "date": "2026-08-17", "close": 101.0, "value": 101.0}
            ],
            "artifact_dir": str(child),
        },
    )
    monkeypatch.setattr(
        tools,
        "predict_parent",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("parent used")),
    )

    result = tools.get_forecast("nvda", horizon=1)

    assert result["status"] == "ok"
    assert result["model_source"] == "child"
    assert result["predictions"][0]["value"] == 101.0
    text = tools.format_forecast_for_prompt(result)
    assert "child-1" in text and "101.0000" in text


def test_get_news_falls_back_to_yahoo(monkeypatch):
    monkeypatch.setattr(tools, "FINNHUB_API_KEY", "fake-key")

    def boom(*args, **kwargs):
        raise RuntimeError("finnhub down")

    monkeypatch.setattr(tools, "_news_from_finnhub", boom)
    monkeypatch.setattr(
        tools,
        "_news_from_yahoo",
        lambda ticker, limit=5: [
            {
                "source": "yahoo",
                "date": "2026-08-15",
                "headline": "NVDA rises",
                "summary": "Chip demand",
                "url": "https://example.com",
            }
        ],
    )

    result = tools.get_news("NVDA")

    assert result["status"] == "ok"
    assert result["provider"] == "yahoo"
    assert "NVDA rises" in tools.format_news_for_prompt(result)
