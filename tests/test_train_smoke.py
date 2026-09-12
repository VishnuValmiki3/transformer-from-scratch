"""End-to-end training smoke tests: tiny model, synthetic corpus, CPU only."""

import json
import os

import numpy as np
import pytest
import torch

from tfs.checkpoint import load_checkpoint
from tfs.data.dataset import TOKEN_DTYPE
from tfs.nn import TransformerLM
from tfs.train import TrainConfig, build_model, estimate_loss, train

VOCAB_SIZE = 64
PERIOD = 17


@pytest.fixture()
def corpus(tmp_path):
    """A periodic token stream — learnable enough that loss must drop within a few steps."""
    pattern = np.arange(PERIOD, dtype=TOKEN_DTYPE) % VOCAB_SIZE
    tokens = np.tile(pattern, 2000)
    path = tmp_path / "train.bin"
    tokens.tofile(path)
    return str(path)


def _config(corpus, out_dir, **overrides) -> TrainConfig:
    base = dict(
        train_tokens=corpus,
        val_tokens=corpus,
        vocab_size=VOCAB_SIZE,
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_layers=1,
        context_length=16,
        lr_max=1e-2,
        lr_min=1e-3,
        warmup_steps=5,
        cosine_steps=60,
        batch_size=8,
        max_steps=60,
        eval_interval=30,
        eval_batches=2,
        checkpoint_interval=0,
        log_interval=1,
        out_dir=out_dir,
        device="cpu",
        amp=False,
    )
    base.update(overrides)
    return TrainConfig(**base)


def test_training_reduces_loss(corpus, tmp_path):
    cfg = _config(corpus, str(tmp_path / "run"))
    train(cfg)

    losses = []
    with open(os.path.join(cfg.out_dir, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if "train_loss" in record:
                losses.append(record["train_loss"])

    assert len(losses) == cfg.max_steps
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 * 0.6


def test_writes_final_checkpoint_that_loads(corpus, tmp_path):
    cfg = _config(corpus, str(tmp_path / "run"))
    final_path = train(cfg)

    assert os.path.exists(final_path)
    model = build_model(cfg, device="cpu")
    assert load_checkpoint(final_path, model, device="cpu") == cfg.max_steps


def test_logs_validation_loss_and_throughput(corpus, tmp_path):
    cfg = _config(corpus, str(tmp_path / "run"))
    train(cfg)

    records = [
        json.loads(line)
        for line in open(os.path.join(cfg.out_dir, "metrics.jsonl"), encoding="utf-8")
    ]
    assert any("val_loss" in r and "val_perplexity" in r for r in records)
    assert all(r["tokens_per_sec"] > 0 for r in records if "tokens_per_sec" in r)


def test_resume_continues_from_saved_step(corpus, tmp_path):
    first = _config(corpus, str(tmp_path / "run1"), max_steps=10)
    checkpoint = train(first)

    second = _config(corpus, str(tmp_path / "run2"), max_steps=20)
    train(second, resume=checkpoint)

    steps = [
        json.loads(line)["step"]
        for line in open(os.path.join(second.out_dir, "metrics.jsonl"), encoding="utf-8")
    ]
    assert min(steps) == 11
    assert max(steps) == 20


def test_learning_rate_follows_the_schedule(corpus, tmp_path):
    cfg = _config(corpus, str(tmp_path / "run"))
    train(cfg)

    lrs = {}
    with open(os.path.join(cfg.out_dir, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if "lr" in record:
                lrs[record["step"]] = record["lr"]

    assert lrs[5] == pytest.approx(cfg.lr_max, rel=1e-6)  # peak at end of warmup
    assert lrs[1] < lrs[5]  # warming up
    assert lrs[60] < lrs[5]  # decaying afterwards


def test_estimate_loss_does_not_leave_model_in_eval_mode(corpus, tmp_path):
    cfg = _config(corpus, str(tmp_path / "run"))
    model = build_model(cfg, device="cpu")
    model.train()
    data = np.fromfile(corpus, dtype=TOKEN_DTYPE)

    estimate_loss(model, data, cfg, "cpu", batches=1)
    assert model.training


def test_config_rejects_unknown_yaml_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("train_tokens: x.bin\nnum_layerz: 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        TrainConfig.from_yaml(str(path))


def test_config_loads_the_shipped_tinystories_yaml():
    cfg = TrainConfig.from_yaml("configs/tinystories_small.yaml")
    assert cfg.vocab_size == 10_000
    assert isinstance(cfg.betas, tuple)
    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        num_layers=cfg.num_layers,
        max_seq_len=cfg.context_length,
    )
    assert 10e6 < model.num_parameters() < 30e6
