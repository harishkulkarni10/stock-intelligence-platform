from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data.ingestion import FEATURE_COLUMNS
from src.pipelines import data_pipeline
from src.pipelines.data_pipeline import build_dataset, dataset_report, load_features


def raw_prices(rows: int = 400) -> pd.DataFrame:
    index = pd.bdate_range("2023-01-02", periods=rows, name="Date")
    close = 100 + np.arange(rows, dtype=float) * 0.2 + np.sin(np.arange(rows) / 5)
    return pd.DataFrame(
        {
            "Open": close - 0.3,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000.0 + np.arange(rows),
        },
        index=index,
    )


@pytest.fixture
def store(tmp_path):
    cfg = Config(feature_path=tmp_path / "features.parquet", workdir=tmp_path / "outputs")
    summary = build_dataset(
        ["^GSPC", "nvda"],
        cfg=cfg,
        downloader=lambda **kwargs: raw_prices(),
    )
    return cfg, summary


def test_build_persists_requested_tickers(store):
    cfg, summary = store

    assert set(summary["tickers"]) == {"^GSPC", "NVDA"}
    stored = pd.read_parquet(cfg.feature_path)
    assert set(stored["ticker"].unique()) == {"^GSPC", "NVDA"}
    assert summary["tickers"]["NVDA"]["rows"] > cfg.context_len + cfg.pred_len


def test_load_features_returns_training_shape(store):
    cfg, _ = store

    frame = load_features("nvda", cfg=cfg)

    assert list(frame.columns) == ["date", *FEATURE_COLUMNS]
    assert frame["date"].is_monotonic_increasing
    assert not frame["date"].duplicated().any()
    assert frame[FEATURE_COLUMNS].notna().all().all()


def test_load_features_reports_missing_inputs(tmp_path):
    cfg = Config(feature_path=tmp_path / "features.parquet")

    with pytest.raises(FileNotFoundError):
        load_features("NVDA", cfg=cfg)


def test_report_covers_splits_and_absent_store(store, tmp_path):
    cfg, _ = store

    report = dataset_report(cfg=cfg)
    nvda = report["tickers"]["NVDA"]

    assert report["exists"] is True
    assert nvda["missing_values"] == 0
    assert nvda["splits"]["train"] > 0
    assert nvda["splits"]["val"] > 0
    assert nvda["splits"]["test"] > 0

    empty = dataset_report(cfg=Config(feature_path=tmp_path / "absent.parquet"))
    assert empty == {"feature_path": str(tmp_path / "absent.parquet"), "exists": False, "tickers": {}}


def test_materialize_requires_data(tmp_path):
    cfg = Config(feature_path=tmp_path / "features.parquet")

    with pytest.raises(FileNotFoundError):
        data_pipeline.materialize_feature_store(cfg=cfg)
