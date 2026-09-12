"""Config-driven pretraining loop for the from-scratch Transformer LM."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml

from .checkpoint import load_checkpoint, save_checkpoint
from .data.dataset import get_batch, load_tokens
from .nn import TransformerLM, cross_entropy
from .optim import AdamW, clip_grad_norm_, cosine_lr


@dataclass
class TrainConfig:
    train_tokens: str
    val_tokens: str | None = None
    vocab_size: int = 10_000

    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 1344
    num_layers: int = 4
    context_length: int = 256
    rope_theta: float = 10_000.0
    tie_embeddings: bool = True

    lr_max: float = 3e-4
    lr_min: float = 3e-5
    warmup_steps: int = 200
    cosine_steps: int = 5_000
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    batch_size: int = 32
    max_steps: int = 5_000
    eval_interval: int = 250
    eval_batches: int = 20
    checkpoint_interval: int = 1_000
    log_interval: int = 10

    out_dir: str = "checkpoints/tinystories"
    device: str = "auto"
    seed: int = 0
    amp: bool = True

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})

        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        if "betas" in raw:
            raw["betas"] = tuple(raw["betas"])
        return cls(**raw)

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(cfg: TrainConfig, device: str | None = None) -> TransformerLM:
    return TransformerLM(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        num_layers=cfg.num_layers,
        max_seq_len=cfg.context_length,
        theta=cfg.rope_theta,
        tie_embeddings=cfg.tie_embeddings,
        device=device or cfg.resolved_device(),
    )


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    data: np.ndarray,
    cfg: TrainConfig,
    device: str,
    batches: int,
    generator: np.random.Generator | None = None,
) -> float:
    model.eval()
    total = 0.0
    for _ in range(batches):
        inputs, targets = get_batch(data, cfg.batch_size, cfg.context_length, device, generator)
        total += cross_entropy(model(inputs), targets).item()
    model.train()
    return total / batches


def train(cfg: TrainConfig, resume: str | None = None) -> str:
    device = cfg.resolved_device()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    eval_rng_seed = cfg.seed + 1  # fixed eval windows so val loss is comparable across steps

    train_data = load_tokens(cfg.train_tokens)
    val_data = load_tokens(cfg.val_tokens) if cfg.val_tokens else None

    model = build_model(cfg, device)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr_max,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    start_step = load_checkpoint(resume, model, optimizer, device) if resume else 0

    use_amp = cfg.amp and device == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    os.makedirs(cfg.out_dir, exist_ok=True)
    log_path = os.path.join(cfg.out_dir, "metrics.jsonl")
    final_path = os.path.join(cfg.out_dir, "final.pt")

    print(
        f"device={device} amp={use_amp} params={model.num_parameters():,} "
        f"train_tokens={len(train_data):,} steps={start_step}->{cfg.max_steps}"
    )

    def log(record: dict) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    model.train()
    window_start = time.perf_counter()
    window_tokens = 0

    for step in range(start_step, cfg.max_steps):
        # Schedule is indexed by steps *completed*, so the first step gets a nonzero
        # warmup LR instead of exactly 0, and logged step N always pairs with cosine_lr(N).
        lr = cosine_lr(step + 1, cfg.lr_max, cfg.lr_min, cfg.warmup_steps, cfg.cosine_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = get_batch(train_data, cfg.batch_size, cfg.context_length, device, rng)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = cross_entropy(model(inputs), targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # clip on true gradients, not scaled ones
        grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        window_tokens += cfg.batch_size * cfg.context_length

        if (step + 1) % cfg.log_interval == 0:
            elapsed = time.perf_counter() - window_start
            tokens_per_sec = window_tokens / elapsed
            record = {
                "step": step + 1,
                "train_loss": loss.item(),
                "lr": lr,
                "grad_norm": grad_norm,
                "tokens_per_sec": tokens_per_sec,
            }
            log(record)
            print(
                f"step {step + 1:>6} loss {loss.item():.4f} lr {lr:.2e} "
                f"|g| {grad_norm:.2f} {tokens_per_sec:,.0f} tok/s"
            )
            window_start = time.perf_counter()
            window_tokens = 0

        if val_data is not None and (step + 1) % cfg.eval_interval == 0:
            val_loss = estimate_loss(
                model, val_data, cfg, device, cfg.eval_batches, np.random.default_rng(eval_rng_seed)
            )
            log({"step": step + 1, "val_loss": val_loss, "val_perplexity": math.exp(val_loss)})
            print(f"step {step + 1:>6} val_loss {val_loss:.4f} ppl {math.exp(val_loss):.2f}")
            window_start = time.perf_counter()
            window_tokens = 0

        if cfg.checkpoint_interval and (step + 1) % cfg.checkpoint_interval == 0:
            save_checkpoint(
                os.path.join(cfg.out_dir, f"step_{step + 1}.pt"),
                model,
                optimizer,
                step + 1,
                dataclasses.asdict(cfg),
            )

    save_checkpoint(final_path, model, optimizer, cfg.max_steps, dataclasses.asdict(cfg))
    print(f"saved {final_path}")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the from-scratch Transformer LM.")
    parser.add_argument("--config", required=True, help="path to a YAML training config")
    parser.add_argument("--resume", help="checkpoint to resume from")
    parser.add_argument("--train-tokens", dest="train_tokens")
    parser.add_argument("--val-tokens", dest="val_tokens")
    parser.add_argument("--out-dir", dest="out_dir")
    parser.add_argument("--device")
    parser.add_argument("--max-steps", dest="max_steps", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k not in {"config", "resume"}}
    train(TrainConfig.from_yaml(args.config, **overrides), resume=args.resume)


if __name__ == "__main__":
    main()
