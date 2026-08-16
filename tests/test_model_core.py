import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.inference import decode_forecast
from src.model.definition import LSTMForecaster
from src.model.evaluation import evaluate_forecast, select_champion
from src.model.saving import load_artifact, save_artifact
from src.model.training import train_model


def test_lstm_output_shape():
    model = LSTMForecaster(3, hidden_size=8, layers=2, horizon=4, dropout=0.2)
    assert model(torch.randn(5, 10, 3)).shape == (5, 4)


def test_training_restores_best_snapshot(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.ones(2, 1)), batch_size=2)
    validation_losses = iter([1.0, 2.0])

    def fake_epoch(model, loader, criterion, device, optimizer=None, max_grad_norm=1.0):
        if optimizer is not None:
            with torch.no_grad():
                model.weight.add_(1.0)
            return 1.0
        return next(validation_losses)

    monkeypatch.setattr("src.model.training._epoch_loss", fake_epoch)
    initial = model.weight.detach().clone()
    result = train_model(model, loader, loader, epochs=2, patience=1)
    assert result["best_epoch"] == 1
    assert result["best_loss"] == 1.0
    assert torch.equal(model.weight, initial + 1.0)


def test_price_metrics_baseline_and_champion():
    metrics = evaluate_forecast(
        np.array([[0.1, 0.2]]),
        np.array([[0.1, -0.1]]),
        np.array([100.0]),
    )
    assert metrics["model"]["mae"] == pytest.approx(15.0)
    assert metrics["persistence"]["rmse"] == pytest.approx(10.0)
    assert metrics["model"]["directional_accuracy"] == 0.5
    candidates = {"persistence": {"mae": 10}, "parent": {"mae": 8}, "child": {"mae": 7.5}}
    assert select_champion(candidates, minimum_child_improvement=0.1) == "parent"
    assert select_champion(candidates, minimum_child_improvement=0.05) == "child"


def test_artifact_roundtrip(tmp_path):
    model = LSTMForecaster(2, hidden_size=4, layers=1, horizon=2, dropout=0.0)
    scaler = StandardScaler().fit([[1.0, 2.0], [3.0, 4.0]])
    metadata = {
        "ticker": "TEST",
        "features": ["close", "volume"],
        "target_mode": "cumulative_return",
        "transform": "simple",
        "lookback": 10,
        "horizon": 2,
        "metrics": {"mae": 1.2},
    }
    save_artifact(tmp_path, model, scaler, metadata)
    loaded, loaded_scaler, loaded_meta = load_artifact(tmp_path)
    inputs = torch.randn(2, 10, 2)
    assert torch.allclose(model(inputs), loaded(inputs))
    assert np.allclose(scaler.transform([[2.0, 3.0]]), loaded_scaler.transform([[2.0, 3.0]]))
    assert loaded_meta["ticker"] == "TEST"
    assert {path.name for path in tmp_path.iterdir()} == {
        "model.pt",
        "scaler.joblib",
        "meta.json",
    }


def test_forecast_decoding_uses_business_days():
    result = decode_forecast([0.01, -0.02], 100.0, "2026-08-14")
    assert result.tolist() == pytest.approx([101.0, 98.0])
    assert result.index.equals(pd.DatetimeIndex(["2026-08-17", "2026-08-18"]))
