"""LoRA: low-rank adapters for parameter-efficient fine-tuning (Hu et al., 2021).

A frozen weight W is adapted as W + (alpha/r) * B @ A, with A and B far smaller than W.
B starts at zero, so a freshly adapted model is numerically identical to the base model —
fine-tuning then departs from the pretrained function smoothly rather than from a jolt.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .linear import Linear

DEFAULT_TARGET_MODULES = ("q_proj", "v_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.base = base
        self.base.weight.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        factory = dict(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, **factory))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, **factory))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = (x @ self.lora_a.T) @ self.lora_b.T
        return self.base(x) + update * self.scaling

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + (self.lora_b @ self.lora_a) * self.scaling


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES,
) -> int:
    """Swap every targeted Linear for a LoRA-wrapped copy. Returns how many were replaced."""
    # Collect first, mutate second: replacing children while walking the tree is unsafe.
    targets = [
        (parent, name, child)
        for parent in model.modules()
        for name, child in parent.named_children()
        if name in target_modules and isinstance(child, Linear)
    ]
    for parent, name, child in targets:
        setattr(parent, name, LoRALinear(child, rank, alpha))
    return len(targets)


def mark_only_lora_trainable(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def lora_parameters(model: nn.Module):
    return [p for name, p in model.named_parameters() if "lora_" in name]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Just the adapter tensors — a few hundred KB instead of the full checkpoint."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


@torch.no_grad()
def merge_lora_weights(model: nn.Module) -> int:
    """Fold adapters back into their base Linear layers, restoring plain Linear modules.

    After merging, inference costs exactly what the base model costs — LoRA's other
    selling point alongside cheap training.
    """
    targets = [
        (parent, name, child)
        for parent in model.modules()
        for name, child in parent.named_children()
        if isinstance(child, LoRALinear)
    ]
    for parent, name, child in targets:
        merged = child.base
        merged.weight.copy_(child.merged_weight())
        merged.weight.requires_grad_(True)
        setattr(parent, name, merged)
    return len(targets)
