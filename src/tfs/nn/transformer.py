"""Pre-norm Transformer block and the decoder-only language model that stacks them."""

from __future__ import annotations

import torch
from torch import nn

from .activations import SwiGLU
from .attention import CausalMultiHeadSelfAttention
from .embedding import Embedding
from .linear import Linear
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm residual block: x + attn(norm(x)), then x + ffn(norm(x))."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_len: int,
        theta: float = 10_000.0,
        tie_embeddings: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (batch, seq_len) -> logits (batch, seq_len, vocab_size)."""
        seq_len = token_ids.shape[-1]
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        token_positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        return self.lm_head(self.ln_final(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
