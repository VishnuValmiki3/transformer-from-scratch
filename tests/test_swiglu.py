import torch
import torch.nn.functional as F

from tfs.nn import SwiGLU, silu


def test_silu_matches_torch_reference():
    x = torch.randn(100)
    assert torch.allclose(silu(x), F.silu(x), atol=1e-6)


def test_silu_is_not_monotonic_near_negative_values():
    """SiLU dips below zero before recovering — the property that distinguishes it from ReLU."""
    x = torch.linspace(-4, 0, 50)
    assert silu(x).min() < -0.2


def test_swiglu_output_shape():
    ffn = SwiGLU(d_model=16, d_ff=64)
    assert ffn(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_swiglu_matches_reference_formula():
    ffn = SwiGLU(d_model=8, d_ff=32)
    x = torch.randn(3, 8)
    expected = ffn.w2(F.silu(ffn.w1(x)) * ffn.w3(x))
    assert torch.allclose(ffn(x), expected, atol=1e-6)


def test_gate_actually_gates():
    """Zeroing the w3 (up) projection must zero the output — proves the multiplicative gate."""
    ffn = SwiGLU(d_model=8, d_ff=16)
    with torch.no_grad():
        ffn.w3.weight.zero_()
    out = ffn(torch.randn(2, 8))
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_gradients_reach_all_three_projections():
    ffn = SwiGLU(d_model=8, d_ff=16)
    ffn(torch.randn(2, 8)).sum().backward()
    for proj in (ffn.w1, ffn.w2, ffn.w3):
        assert proj.weight.grad is not None
        assert proj.weight.grad.abs().sum() > 0
