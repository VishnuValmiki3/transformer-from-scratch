"""Checkpoint save/load for resumable training runs."""

from __future__ import annotations

import os
from typing import Any

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: dict[str, Any] | None = None,
) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> int:
    """Restore model (and optionally optimizer) in place; returns the saved step."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["step"]


def load_config(path: str, device: str | torch.device = "cpu") -> dict[str, Any] | None:
    """Read just the config a checkpoint was trained with, without building a model."""
    return torch.load(path, map_location=device, weights_only=False).get("config")
