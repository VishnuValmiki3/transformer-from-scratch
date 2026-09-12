import math

import torch

from tfs.nn import Linear


def test_output_shape_and_no_bias():
    layer = Linear(16, 32)
    out = layer(torch.randn(4, 7, 16))
    assert out.shape == (4, 7, 32)
    assert not hasattr(layer, "bias") or layer.bias is None


def test_matches_manual_matmul():
    layer = Linear(8, 5)
    x = torch.randn(3, 8)
    assert torch.allclose(layer(x), x @ layer.weight.T)


def test_init_std_matches_fan_in_fan_out_rule():
    in_features, out_features = 512, 512
    layer = Linear(in_features, out_features)
    expected_std = math.sqrt(2.0 / (in_features + out_features))
    assert abs(layer.weight.std().item() - expected_std) / expected_std < 0.1


def test_init_is_truncated_at_three_std():
    in_features, out_features = 256, 256
    layer = Linear(in_features, out_features)
    std = math.sqrt(2.0 / (in_features + out_features))
    assert layer.weight.abs().max().item() <= 3 * std + 1e-6


def test_gradients_flow_to_weight():
    layer = Linear(8, 4)
    layer(torch.randn(2, 8)).sum().backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
