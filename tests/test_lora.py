import pytest
import torch

from tfs.nn import (
    Linear,
    LoRALinear,
    TransformerLM,
    inject_lora,
    lora_parameters,
    lora_state_dict,
    mark_only_lora_trainable,
    merge_lora_weights,
)

MODEL_KWARGS = dict(
    vocab_size=50, d_model=32, num_heads=4, d_ff=64, num_layers=2, max_seq_len=16
)


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return TransformerLM(**MODEL_KWARGS)


def test_adapter_starts_as_a_no_op():
    """B is zero-initialized, so an untrained adapter must reproduce the base layer exactly."""
    base = Linear(16, 24)
    x = torch.randn(3, 16)
    expected = base(x)
    assert torch.allclose(LoRALinear(base, rank=4, alpha=8.0)(x), expected, atol=1e-6)


def test_adapter_changes_output_once_b_is_nonzero():
    base = Linear(16, 24)
    layer = LoRALinear(base, rank=4, alpha=8.0)
    with torch.no_grad():
        layer.lora_b.normal_()
    x = torch.randn(3, 16)
    assert not torch.allclose(layer(x), base(x), atol=1e-4)


def test_scaling_is_alpha_over_rank():
    base = Linear(8, 8)
    layer = LoRALinear(base, rank=4, alpha=16.0)
    assert layer.scaling == 4.0

    with torch.no_grad():
        layer.lora_a.fill_(1.0)
        layer.lora_b.fill_(1.0)
    x = torch.ones(1, 8)
    # update = (x @ A^T) @ B^T * scaling = (8 * 4) per output unit * 4
    assert torch.allclose(layer(x) - base(x), torch.full((1, 8), 128.0), atol=1e-4)


def test_rejects_nonpositive_rank():
    with pytest.raises(ValueError):
        LoRALinear(Linear(4, 4), rank=0, alpha=1.0)


def test_base_weight_is_frozen_but_adapters_are_not():
    layer = LoRALinear(Linear(8, 8), rank=2, alpha=4.0)
    assert not layer.base.weight.requires_grad
    assert layer.lora_a.requires_grad and layer.lora_b.requires_grad


def test_gradients_reach_adapters_only():
    layer = LoRALinear(Linear(8, 8), rank=2, alpha=4.0)
    with torch.no_grad():
        layer.lora_b.normal_()
    layer(torch.randn(2, 8)).sum().backward()

    assert layer.base.weight.grad is None
    assert layer.lora_a.grad is not None and layer.lora_a.grad.abs().sum() > 0
    assert layer.lora_b.grad is not None and layer.lora_b.grad.abs().sum() > 0


def test_injection_wraps_only_targeted_modules(model):
    replaced = inject_lora(model, rank=4, alpha=8.0, target_modules=("q_proj", "v_proj"))
    assert replaced == 2 * MODEL_KWARGS["num_layers"]

    for block in model.layers:
        assert isinstance(block.attn.q_proj, LoRALinear)
        assert isinstance(block.attn.v_proj, LoRALinear)
        assert isinstance(block.attn.k_proj, Linear)
        assert not isinstance(block.attn.k_proj, LoRALinear)


def test_injection_does_not_change_model_output(model):
    tokens = torch.randint(0, 50, (2, 8))
    baseline = model(tokens)
    inject_lora(model, rank=4, alpha=8.0)
    assert torch.allclose(model(tokens), baseline, atol=1e-6)


def test_injection_is_not_applied_twice(model):
    inject_lora(model, rank=4, alpha=8.0)
    assert inject_lora(model, rank=4, alpha=8.0) == 0


def test_mark_only_lora_trainable_freezes_everything_else(model):
    inject_lora(model, rank=4, alpha=8.0)
    mark_only_lora_trainable(model)

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all("lora_" in name for name in trainable)


def test_trainable_fraction_is_tiny(model):
    inject_lora(model, rank=4, alpha=8.0)
    mark_only_lora_trainable(model)

    trainable = sum(p.numel() for p in lora_parameters(model))
    total = sum(p.numel() for p in model.parameters())
    assert trainable / total < 0.05


def test_adapter_state_dict_holds_only_adapters(model):
    inject_lora(model, rank=4, alpha=8.0)
    state = lora_state_dict(model)
    assert state
    assert all("lora_" in key for key in state)
    assert all(key in model.state_dict() for key in state)


def test_merging_preserves_outputs_and_restores_plain_linears(model):
    tokens = torch.randint(0, 50, (2, 8))
    inject_lora(model, rank=4, alpha=8.0)
    for param in lora_parameters(model):
        with torch.no_grad():
            param.normal_(std=0.02)

    model.eval()
    before = model(tokens)
    merged = merge_lora_weights(model)
    after = model(tokens)

    assert merged == 2 * MODEL_KWARGS["num_layers"]
    assert torch.allclose(before, after, atol=1e-5)
    for block in model.layers:
        assert isinstance(block.attn.q_proj, Linear)
        assert not isinstance(block.attn.q_proj, LoRALinear)


def test_optimizer_step_moves_adapters_but_not_base(model):
    inject_lora(model, rank=4, alpha=8.0)
    mark_only_lora_trainable(model)

    base_before = model.layers[0].attn.q_proj.base.weight.clone()
    adapter_before = model.layers[0].attn.q_proj.lora_b.clone()

    optimizer = torch.optim.SGD(lora_parameters(model), lr=0.1)
    model(torch.randint(0, 50, (2, 8))).sum().backward()
    optimizer.step()

    assert torch.equal(model.layers[0].attn.q_proj.base.weight, base_before)
    assert not torch.equal(model.layers[0].attn.q_proj.lora_b, adapter_before)
