"""Deterministic training utilities."""

from __future__ import annotations

import copy
import random
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import nn


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _epoch_loss(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float = 1.0,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predictions = model(inputs)
            loss = criterion(predictions, targets)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        items = inputs.shape[0]
        total_loss += loss.item() * items
        total_items += items

    if not total_items:
        raise ValueError("data loader is empty")
    return total_loss / total_items


def train_model(
    model: nn.Module,
    train_loader: Iterable,
    validation_loader: Iterable,
    *,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    patience: int = 10,
    min_delta: float = 0.0,
    scheduler_patience: int = 3,
    scheduler_factor: float = 0.5,
    seed: int = 42,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train ``model``, restore its best validation snapshot, and return diagnostics."""
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    set_deterministic_seed(seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "validation_loss": [],
        "learning_rate": [],
    }
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = _epoch_loss(
            model, train_loader, criterion, target_device, optimizer, max_grad_norm
        )
        with torch.no_grad():
            validation_loss = _epoch_loss(
                model, validation_loader, criterion, target_device
            )
        scheduler.step(validation_loss)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:  # Defensive: a finite or non-finite first loss still gets restored.
        best_state = copy.deepcopy(model.state_dict())
        best_loss = history["validation_loss"][0]
        best_epoch = 1
    model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "best_loss": best_loss}


train = train_model
