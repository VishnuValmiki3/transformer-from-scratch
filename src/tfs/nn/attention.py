"""Softmax, scaled dot-product attention, and causal multi-head self-attention.

Everything here is written against raw tensor ops — no nn.MultiheadAttention,
no F.scaled_dot_product_attention, no F.softmax.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .linear import Linear
from .positional import RotaryPositionalEmbedding


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Subtract the max for numerical stability: exp() of large logits overflows otherwise.
    shifted = x - x.amax(dim=dim, keepdim=True)
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """q/k/v: (..., seq_len, d_k). mask: bool tensor, True = attend, False = block."""
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return softmax(scores, dim=-1) @ v


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = (
            RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
            if theta is not None and max_seq_len is not None
            else None
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        attended = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        merged = attended.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.output_proj(merged)
