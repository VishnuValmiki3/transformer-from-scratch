import torch

from tfs.checkpoint import load_checkpoint, load_config, save_checkpoint
from tfs.nn import TransformerLM
from tfs.optim import AdamW

MODEL_KWARGS = dict(
    vocab_size=40, d_model=16, num_heads=2, d_ff=32, num_layers=1, max_seq_len=8
)


def _model_and_optimizer():
    torch.manual_seed(0)
    model = TransformerLM(**MODEL_KWARGS)
    return model, AdamW(model.parameters(), lr=1e-3)


def test_restores_model_weights(tmp_path):
    model, optimizer = _model_and_optimizer()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, optimizer, step=7)

    restored = TransformerLM(**MODEL_KWARGS)
    load_checkpoint(path, restored)

    for (name, saved), (_, loaded) in zip(model.named_parameters(), restored.named_parameters()):
        assert torch.equal(saved, loaded), name


def test_returns_saved_step(tmp_path):
    model, optimizer = _model_and_optimizer()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, optimizer, step=123)
    assert load_checkpoint(path, TransformerLM(**MODEL_KWARGS)) == 123


def test_restores_optimizer_moment_estimates(tmp_path):
    """Resuming without optimizer state would restart Adam's moments and spike the loss."""
    model, optimizer = _model_and_optimizer()
    model(torch.randint(0, 40, (2, 8))).sum().backward()
    optimizer.step()

    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, optimizer, step=1)

    fresh_model, fresh_optimizer = _model_and_optimizer()
    load_checkpoint(path, fresh_model, fresh_optimizer)

    original_state = next(iter(optimizer.state.values()))
    restored_state = next(iter(fresh_optimizer.state.values()))
    assert restored_state["step"] == original_state["step"]
    assert torch.allclose(restored_state["exp_avg"], original_state["exp_avg"])


def test_model_only_load_ignores_missing_optimizer(tmp_path):
    model, _ = _model_and_optimizer()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, optimizer=None, step=3)
    assert load_checkpoint(path, TransformerLM(**MODEL_KWARGS)) == 3


def test_config_roundtrips_with_the_weights(tmp_path):
    model, optimizer = _model_and_optimizer()
    path = str(tmp_path / "ckpt.pt")
    config = {"d_model": 16, "note": "smoke"}
    save_checkpoint(path, model, optimizer, step=1, config=config)
    assert load_config(path) == config


def test_creates_missing_output_directory(tmp_path):
    model, optimizer = _model_and_optimizer()
    path = str(tmp_path / "nested" / "dir" / "ckpt.pt")
    save_checkpoint(path, model, optimizer, step=1)
    assert load_checkpoint(path, TransformerLM(**MODEL_KWARGS)) == 1
