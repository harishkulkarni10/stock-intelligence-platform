"""Price-space evaluation and champion selection."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def returns_to_prices(
    cumulative_returns: np.ndarray,
    anchors: np.ndarray | float,
    *,
    transform: str = "simple",
) -> np.ndarray:
    """Decode direct cumulative-return targets into prices."""
    returns = np.asarray(cumulative_returns, dtype=float)
    anchor = np.asarray(anchors, dtype=float)
    if returns.ndim == 2 and anchor.ndim == 1:
        anchor = anchor[:, None]
    if transform == "simple":
        return anchor * (1.0 + returns)
    if transform == "log":
        return anchor * np.exp(returns)
    raise ValueError("transform must be 'simple' or 'log'")


def price_metrics(
    predicted_prices: np.ndarray,
    actual_prices: np.ndarray,
    anchors: np.ndarray | float,
) -> dict[str, float]:
    predicted = np.asarray(predicted_prices, dtype=float)
    actual = np.asarray(actual_prices, dtype=float)
    if predicted.shape != actual.shape:
        raise ValueError("predicted and actual prices must have matching shapes")
    anchor = np.asarray(anchors, dtype=float)
    if predicted.ndim == 2 and anchor.ndim == 1:
        anchor = anchor[:, None]
    errors = predicted - actual
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape": float(
            np.mean(np.abs(errors) / np.maximum(np.abs(actual), np.finfo(float).eps))
            * 100.0
        ),
        "directional_accuracy": float(
            np.mean(np.sign(predicted - anchor) == np.sign(actual - anchor))
        ),
    }


def evaluate_forecast(
    predicted_returns: np.ndarray,
    actual_returns: np.ndarray,
    anchors: np.ndarray | float,
    *,
    transform: str = "simple",
) -> dict[str, dict[str, float]]:
    """Evaluate a forecast and a zero-return persistence baseline in price space."""
    predicted_prices = returns_to_prices(predicted_returns, anchors, transform=transform)
    actual_prices = returns_to_prices(actual_returns, anchors, transform=transform)
    persistence_prices = returns_to_prices(
        np.zeros_like(actual_returns, dtype=float), anchors, transform=transform
    )
    return {
        "model": price_metrics(predicted_prices, actual_prices, anchors),
        "persistence": price_metrics(persistence_prices, actual_prices, anchors),
    }


def select_champion(
    metrics: Mapping[str, Mapping[str, float] | float],
    *,
    minimum_child_improvement: float = 0.0,
) -> str:
    """Choose the lowest-MAE incumbent, requiring a margin before promoting child."""
    if minimum_child_improvement < 0:
        raise ValueError("minimum_child_improvement cannot be negative")

    def mae(name: str) -> float:
        value = metrics[name]
        return float(value["mae"] if isinstance(value, Mapping) else value)

    incumbents = [name for name in ("persistence", "parent") if name in metrics]
    if not incumbents:
        if "child" in metrics:
            return "child"
        raise ValueError("metrics must include persistence, parent, or child")
    champion = min(incumbents, key=mae)
    if "child" in metrics:
        required_mae = mae(champion) * (1.0 - minimum_child_improvement)
        if mae("child") <= required_mae:
            champion = "child"
    return champion


evaluate = evaluate_forecast
