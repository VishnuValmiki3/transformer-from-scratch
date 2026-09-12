"""Numerically stable cross-entropy, computed without F.cross_entropy / F.log_softmax."""

from __future__ import annotations

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood.

    logits: (..., vocab_size), targets: (...) of integer class indices.

    Computed as logsumexp(logits) - logits[target] so the softmax numerator and
    denominator never have to be materialized — exp() of a raw logit would overflow.
    """
    logits = logits - logits.amax(dim=-1, keepdim=True)
    log_sum_exp = torch.log(torch.exp(logits).sum(dim=-1))
    target_logits = torch.gather(logits, -1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logits).mean()


def perplexity(loss: torch.Tensor | float) -> float:
    value = loss.item() if isinstance(loss, torch.Tensor) else loss
    return float(torch.exp(torch.tensor(value)))
