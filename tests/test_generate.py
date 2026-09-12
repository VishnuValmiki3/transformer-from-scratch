import numpy as np
import pytest
import torch

from tfs.checkpoint import save_checkpoint
from tfs.data.dataset import TOKEN_DTYPE
from tfs.finetune_lora import LoRAConfig, finetune
from tfs.generate import generate, load_model_for_inference, top_p_filter
from tfs.tokenizer.bpe import Tokenizer, train_bpe
from tfs.train import TrainConfig, build_model

VOCAB_SIZE = 300


@pytest.fixture()
def tokenizer(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog. " * 50, encoding="utf-8")
    vocab, merges = train_bpe(str(corpus), VOCAB_SIZE, special_tokens=["<|endoftext|>"])
    return Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])


@pytest.fixture()
def checkpoint(tmp_path, tokenizer):
    cfg = TrainConfig(
        train_tokens="unused.bin",
        vocab_size=len(tokenizer.vocab),
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_layers=1,
        context_length=16,
        device="cpu",
    )
    model = build_model(cfg, "cpu")
    path = str(tmp_path / "base.pt")
    save_checkpoint(path, model, optimizer=None, step=0, config=cfg.__dict__)
    return path


def test_top_p_keeps_only_the_nucleus():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    filtered = top_p_filter(probs, top_p=0.8)
    assert filtered[0] > 0 and filtered[1] > 0
    assert filtered[2] == 0 and filtered[3] == 0
    assert torch.isclose(filtered.sum(), torch.tensor(1.0))


def test_top_p_always_keeps_the_top_token():
    """Even when one token alone exceeds top_p, something must remain sampleable."""
    probs = torch.tensor([0.99, 0.005, 0.005])
    filtered = top_p_filter(probs, top_p=0.5)
    assert filtered[0] == pytest.approx(1.0)


def test_top_p_disabled_returns_probs_unchanged():
    probs = torch.tensor([0.5, 0.3, 0.2])
    assert torch.equal(top_p_filter(probs, top_p=1.0), probs)


def test_generate_extends_the_prompt(tokenizer, checkpoint):
    model = load_model_for_inference(checkpoint, "cpu")
    text = generate(model, tokenizer, "the quick", max_new_tokens=15, device="cpu")
    assert text.startswith("the quick")
    assert len(text) > len("the quick")


def test_generate_is_deterministic_at_temperature_zero(tokenizer, checkpoint):
    model = load_model_for_inference(checkpoint, "cpu")
    kwargs = dict(max_new_tokens=10, temperature=0.0, device="cpu")
    assert generate(model, tokenizer, "the", **kwargs) == generate(
        model, tokenizer, "the", **kwargs
    )


def test_generate_is_reproducible_with_a_seeded_generator(tokenizer, checkpoint):
    model = load_model_for_inference(checkpoint, "cpu")

    def sample():
        return generate(
            model,
            tokenizer,
            "the",
            max_new_tokens=10,
            device="cpu",
            generator=torch.Generator().manual_seed(7),
        )

    assert sample() == sample()


def test_generate_stops_at_eos(tokenizer, checkpoint):
    model = load_model_for_inference(checkpoint, "cpu")
    eos_id = tokenizer.byte_to_id["<|endoftext|>".encode("utf-8")]

    # Force the model to emit EOS immediately by making that logit dominate.
    with torch.no_grad():
        model.lm_head.weight[eos_id] += 1e4

    text = generate(model, tokenizer, "the", max_new_tokens=50, device="cpu", eos_id=eos_id)
    assert "<|endoftext|>" not in text


def test_generate_respects_the_context_window(tokenizer, checkpoint):
    """Prompts longer than max_seq_len must be truncated, not crash the forward pass."""
    model = load_model_for_inference(checkpoint, "cpu")
    long_prompt = "the quick brown fox jumps over the lazy dog. " * 10
    assert generate(model, tokenizer, long_prompt, max_new_tokens=5, device="cpu")


def test_finetune_produces_a_loadable_adapter(tmp_path, checkpoint, tokenizer):
    tokens = (
        np.random.default_rng(0).integers(0, len(tokenizer.vocab), size=5000).astype(TOKEN_DTYPE)
    )
    token_path = tmp_path / "style.bin"
    tokens.tofile(token_path)

    cfg = LoRAConfig(
        base_checkpoint=checkpoint,
        train_tokens=str(token_path),
        val_tokens=str(token_path),
        rank=4,
        alpha=8.0,
        max_steps=5,
        warmup_steps=1,
        cosine_steps=5,
        batch_size=4,
        eval_interval=5,
        eval_batches=1,
        log_interval=1,
        out_dir=str(tmp_path / "lora"),
        device="cpu",
        amp=False,
    )
    adapter_path = finetune(cfg)

    payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    assert payload["rank"] == 4
    assert all("lora_" in key for key in payload["lora"])

    adapted = load_model_for_inference(checkpoint, "cpu", adapter=adapter_path)
    assert generate(adapted, tokenizer, "the", max_new_tokens=5, device="cpu")


def test_adapted_model_differs_from_base(tmp_path, checkpoint, tokenizer):
    """A trained adapter must actually change the model's predictions."""
    tokens = np.tile(np.arange(23, dtype=TOKEN_DTYPE), 300)
    token_path = tmp_path / "style.bin"
    tokens.tofile(token_path)

    cfg = LoRAConfig(
        base_checkpoint=checkpoint,
        train_tokens=str(token_path),
        rank=4,
        alpha=8.0,
        lr_max=1e-2,
        max_steps=30,
        warmup_steps=2,
        cosine_steps=30,
        batch_size=4,
        eval_interval=1000,
        log_interval=10,
        out_dir=str(tmp_path / "lora"),
        device="cpu",
        amp=False,
    )
    adapter_path = finetune(cfg)

    prompt_ids = torch.tensor([[1, 2, 3, 4]])
    base = load_model_for_inference(checkpoint, "cpu")
    adapted = load_model_for_inference(checkpoint, "cpu", adapter=adapter_path)

    base.eval()
    adapted.eval()
    with torch.no_grad():
        assert not torch.allclose(base(prompt_ids), adapted(prompt_ids), atol=1e-3)
