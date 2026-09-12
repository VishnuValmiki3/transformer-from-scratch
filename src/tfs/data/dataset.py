"""Memmap-backed token dataset and random batch sampling for language-model training."""

from __future__ import annotations

import numpy as np
import torch

# uint16 holds vocabularies up to 65,536 tokens and halves the on-disk/page-cache
# footprint versus int32, which matters when the corpus is hundreds of millions of tokens.
TOKEN_DTYPE = np.uint16


def load_tokens(path: str) -> np.ndarray:
    """Open a .bin token file without reading it into RAM."""
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
    generator: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random windows; targets are inputs shifted one token to the right."""
    if len(data) < context_length + 1:
        raise ValueError(
            f"token file holds {len(data)} tokens, need at least {context_length + 1}"
        )

    rng = generator if generator is not None else np.random.default_rng()
    starts = rng.integers(0, len(data) - context_length, size=batch_size)

    windows = np.stack([data[s : s + context_length + 1] for s in starts]).astype(np.int64)
    batch = torch.from_numpy(windows)

    inputs = batch[:, :-1].to(device)
    targets = batch[:, 1:].to(device)
    return inputs, targets
