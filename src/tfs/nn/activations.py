"""SiLU activation and the SwiGLU feed-forward block (Shazeer, 2020)."""

from __future__ import annotations

import torch
from torch import nn

from .linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """FFN(x) = W2 @ (SiLU(W1 @ x) * (W3 @ x)) — gated feed-forward, no biases."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
