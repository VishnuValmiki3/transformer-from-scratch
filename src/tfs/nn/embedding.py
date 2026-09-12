"""Token embedding lookup built from a raw nn.Parameter (no nn.Embedding)."""

from __future__ import annotations

import torch
from torch import nn


# GPT-2's embedding scale. This is small for a reason: under weight tying the embedding
# matrix is also the output projection, so logits are a d_model-long dot product against
# it. At std=1 that puts logit std near sqrt(d_model) (~22.6 at d_model=512) and starts
# training at a loss of ~342 instead of ln(vocab_size). RMSNorm rescales the input path,
# so the small init costs nothing there.
EMBEDDING_INIT_STD = 0.02


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        std = EMBEDDING_INIT_STD
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
