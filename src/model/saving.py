"""Versioned model artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn

from src.model.definition import LSTMForecaster

REQUIRED_METADATA = {
    "ticker",
    "model_type",
    "features",
    "target_mode",
    "transform",
    "lookback",
    "horizon",
    "config",
    "metrics",
    "version",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def save_artifact(
    directory: str | Path,
    model: nn.Module,
    scaler: Any,
    metadata: dict[str, Any],
) -> Path:
    """Save model weights, fitted scaler, and human-readable metadata."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata)
    meta.setdefault("model_type", type(model).__name__)
    meta.setdefault("config", getattr(model, "config", {}))
    meta.setdefault("version", "1")
    if "horizon" not in meta and "horizon" in meta["config"]:
        meta["horizon"] = meta["config"]["horizon"]
    missing = REQUIRED_METADATA - meta.keys()
    if missing:
        raise ValueError(f"missing artifact metadata: {', '.join(sorted(missing))}")

    torch.save(model.state_dict(), path / "model.pt")
    joblib.dump(scaler, path / "scaler.joblib")
    (path / "meta.json").write_text(
        json.dumps(meta, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def load_artifact(
    directory: str | Path,
    *,
    model_class: type[nn.Module] = LSTMForecaster,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load an artifact and reconstruct an evaluation-ready model."""
    path = Path(directory)
    metadata = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    missing = REQUIRED_METADATA - metadata.keys()
    if missing:
        raise ValueError(f"invalid artifact metadata: missing {', '.join(sorted(missing))}")
    model = model_class(**metadata["config"])
    try:
        state = torch.load(path / "model.pt", map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility.
        state = torch.load(path / "model.pt", map_location=map_location)
    model.load_state_dict(state)
    model.to(map_location)
    model.eval()
    return model, joblib.load(path / "scaler.joblib"), metadata


save_model_artifact = save_artifact
load_model_artifact = load_artifact
