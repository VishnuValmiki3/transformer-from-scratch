"""Cosine learning-rate schedule with linear warmup, plus global gradient clipping."""

from __future__ import annotations

import math
from typing import Iterable

import torch


def cosine_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    cosine_steps: int,
) -> float:
    """Linear warmup to max_lr, cosine decay to min_lr by `cosine_steps`, flat after."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step <= cosine_steps:
        progress = (step - warmup_steps) / max(1, cosine_steps - warmup_steps)
        return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)
    return min_lr


def clip_grad_norm_(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = 1e-6,
) -> float:
    """Rescale gradients in place so their global L2 norm is at most max_norm.

    Returns the pre-clipping norm — worth logging, since a spiking grad norm is the
    earliest visible sign of a diverging run.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)
    return total_norm.item()
