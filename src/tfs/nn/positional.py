"""Rotary Positional Embedding (RoPE) — Su et al., 2021.

Rotates each consecutive (2i, 2i+1) coordinate pair of Q/K by an angle that grows with
absolute position and shrinks with pair index, so dot products naturally encode relative
position without any additive positional embedding.
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension, got d_k={d_k}")

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        angles = torch.outer(positions, inv_freq)  # (max_seq_len, d_k / 2)

        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """x: (..., seq_len, d_k); token_positions: (seq_len,) integer positions into the cache."""
        cos = self.cos_cached[token_positions]  # (seq_len, d_k / 2), broadcasts over leading dims
        sin = self.sin_cached[token_positions]

        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        out = torch.stack([rotated_even, rotated_odd], dim=-1)
        return out.flatten(-2)
