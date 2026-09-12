import torch

from tfs.nn import RMSNorm


def test_preserves_shape():
    norm = RMSNorm(16)
    assert norm(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_output_has_unit_rms_with_default_weight():
    norm = RMSNorm(64, eps=1e-8)
    out = norm(torch.randn(4, 64) * 10)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_is_scale_invariant():
    """The defining property: RMSNorm(c*x) == RMSNorm(x) for positive scalar c."""
    norm = RMSNorm(32, eps=1e-8)
    x = torch.randn(3, 32)
    assert torch.allclose(norm(x), norm(x * 7.5), atol=1e-4)


def test_does_not_center_the_mean():
    """Unlike LayerNorm, RMSNorm must leave a nonzero mean nonzero."""
    norm = RMSNorm(32, eps=1e-8)
    x = torch.randn(3, 32) + 5.0
    assert norm(x).mean(dim=-1).abs().min() > 0.1


def test_learnable_weight_scales_each_channel():
    norm = RMSNorm(8, eps=1e-8)
    x = torch.randn(2, 8)
    baseline = norm(x)
    with torch.no_grad():
        norm.weight.mul_(3.0)
    assert torch.allclose(norm(x), baseline * 3.0, atol=1e-5)


def test_matches_reference_formula():
    d_model, eps = 16, 1e-5
    norm = RMSNorm(d_model, eps=eps)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(d_model))
    x = torch.randn(3, d_model)
    expected = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * norm.weight
    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_upcasts_then_restores_input_dtype():
    norm = RMSNorm(8).to(torch.float16)
    out = norm(torch.randn(2, 8, dtype=torch.float16))
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
