from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data.ingestion import FEATURE_COLUMNS, persist_features
from src.pipelines import inference_pipeline, training_pipeline


def _synthetic_frame(rows: int = 260) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=rows)
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, size=rows))
    frame = pd.DataFrame(
        {
            "date": dates,
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.full(rows, 1_000_000.0),
            "RSI14": np.full(rows, 50.0),
            "MACD": np.zeros(rows),
        }
    )
    return frame.loc[:, ["date", *FEATURE_COLUMNS]]


@pytest.fixture
def pipeline_env(monkeypatch, tmp_path: Path):
    cfg = Config(
        parent_epochs=1,
        child_epochs=1,
        context_len=20,
        pred_len=3,
        batch_size=16,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        parent_dir=tmp_path / "outputs" / "parent",
        workdir=tmp_path / "outputs",
        feature_path=tmp_path / "feature_store" / "data" / "features.parquet",
        device="cpu",
        transfer_strategy="full",
        fine_tune_lr=1e-3,
    )
    monkeypatch.setattr(training_pipeline, "_cfg", lambda: cfg)
    monkeypatch.setattr(inference_pipeline, "_cfg", lambda: cfg)
    monkeypatch.setattr(training_pipeline, "setup_mlflow", lambda: "sqlite:///:memory:")
    frame = _synthetic_frame()
    for ticker in (cfg.parent_ticker, "NVDA"):
        persist_features(frame, ticker, cfg.feature_path)
    monkeypatch.setattr(
        inference_pipeline,
        "fetch_ohlcv",
        lambda ticker, **kwargs: frame.copy(),
    )
    return cfg


def test_parent_train_and_predict(pipeline_env):
    summary = training_pipeline.train_parent(epochs=1, source="feature-store")
    assert (Path(summary["artifact_dir"]) / "model.pt").exists()
    prediction = inference_pipeline.predict_parent("NVDA", horizon=2)
    assert prediction["horizon"] == 2
    assert len(prediction["predictions"]) == 2
    assert prediction["predictions"][0]["close"] > 0


def test_child_train_requires_parent(pipeline_env):
    with pytest.raises(FileNotFoundError):
        training_pipeline.train_child("NVDA", epochs=1, source="feature-store")
    training_pipeline.train_parent(epochs=1, source="feature-store")
    summary = training_pipeline.train_child(
        "NVDA", epochs=1, child_improvement=0.0, source="feature-store"
    )
    assert summary["champion"] in {"child", "parent", "persistence"}


def test_training_rejects_absent_feature_store(pipeline_env):
    pipeline_env.feature_path.unlink()

    with pytest.raises(FileNotFoundError, match="sip-data build"):
        training_pipeline.train_parent(epochs=1, source="feature-store")
