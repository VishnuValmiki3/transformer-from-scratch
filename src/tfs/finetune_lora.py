"""LoRA fine-tuning: adapt a pretrained checkpoint to a new domain by training
a few hundred thousand adapter parameters instead of the whole model."""

from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass

import torch
import yaml

from .checkpoint import load_checkpoint, load_config
from .nn import (
    inject_lora,
    lora_parameters,
    lora_state_dict,
    mark_only_lora_trainable,
)
from .nn.lora import DEFAULT_TARGET_MODULES
from .optim import AdamW
from .train import TrainConfig, build_model, train


@dataclass
class LoRAConfig:
    base_checkpoint: str
    train_tokens: str
    val_tokens: str | None = None

    rank: int = 8
    alpha: float = 16.0
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES

    lr_max: float = 1e-3
    lr_min: float = 1e-4
    warmup_steps: int = 50
    cosine_steps: int = 1_000
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    batch_size: int = 16
    max_steps: int = 1_000
    eval_interval: int = 100
    eval_batches: int = 10
    checkpoint_interval: int = 0
    log_interval: int = 10

    out_dir: str = "checkpoints/lora_style"
    device: str = "auto"
    seed: int = 0
    amp: bool = True

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> "LoRAConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})

        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        for key in ("betas", "target_modules"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)


def load_base_model(base_checkpoint: str, device: str):
    saved = load_config(base_checkpoint, device=device)
    if saved is None:
        raise ValueError(f"{base_checkpoint} has no embedded config; cannot rebuild the model")
    base_cfg = TrainConfig(**saved)
    model = build_model(base_cfg, device)
    load_checkpoint(base_checkpoint, model, device=device)
    return model, base_cfg


def finetune(cfg: LoRAConfig) -> str:
    device = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else cfg.device
    device = "cpu" if device == "auto" else device

    model, base_cfg = load_base_model(cfg.base_checkpoint, device)

    replaced = inject_lora(model, cfg.rank, cfg.alpha, tuple(cfg.target_modules))
    if replaced == 0:
        raise ValueError(f"no modules matched target_modules={cfg.target_modules}")
    mark_only_lora_trainable(model)
    model.to(device)

    trainable = sum(p.numel() for p in lora_parameters(model))
    total = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA rank={cfg.rank} alpha={cfg.alpha} wrapped {replaced} modules\n"
        f"  trainable {trainable:,} / {total:,} params ({100 * trainable / total:.2f}%)"
    )

    optimizer = AdamW(
        lora_parameters(model),
        lr=cfg.lr_max,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    train_cfg = dataclasses.replace(
        base_cfg,
        train_tokens=cfg.train_tokens,
        val_tokens=cfg.val_tokens,
        lr_max=cfg.lr_max,
        lr_min=cfg.lr_min,
        warmup_steps=cfg.warmup_steps,
        cosine_steps=cfg.cosine_steps,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
        grad_clip=cfg.grad_clip,
        batch_size=cfg.batch_size,
        max_steps=cfg.max_steps,
        eval_interval=cfg.eval_interval,
        eval_batches=cfg.eval_batches,
        checkpoint_interval=cfg.checkpoint_interval,
        log_interval=cfg.log_interval,
        out_dir=cfg.out_dir,
        device=device,
        seed=cfg.seed,
        amp=cfg.amp,
    )

    train(train_cfg, model=model, optimizer=optimizer)

    adapter_path = os.path.join(cfg.out_dir, "adapter.pt")
    torch.save(
        {
            "lora": lora_state_dict(model),
            "rank": cfg.rank,
            "alpha": cfg.alpha,
            "target_modules": list(cfg.target_modules),
            "base_checkpoint": cfg.base_checkpoint,
        },
        adapter_path,
    )
    size_kb = os.path.getsize(adapter_path) / 1024
    print(f"saved adapter {adapter_path} ({size_kb:,.1f} KB)")
    return adapter_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-checkpoint", dest="base_checkpoint")
    parser.add_argument("--train-tokens", dest="train_tokens")
    parser.add_argument("--val-tokens", dest="val_tokens")
    parser.add_argument("--out-dir", dest="out_dir")
    parser.add_argument("--device")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--max-steps", dest="max_steps", type=int)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    finetune(LoRAConfig.from_yaml(args.config, **overrides))


if __name__ == "__main__":
    main()
