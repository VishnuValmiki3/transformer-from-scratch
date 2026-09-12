import math

import torch

from tfs.nn import CausalMultiHeadSelfAttention, scaled_dot_product_attention, softmax


def test_softmax_matches_torch_reference():
    x = torch.randn(4, 9)
    assert torch.allclose(softmax(x, dim=-1), torch.softmax(x, dim=-1), atol=1e-6)


def test_softmax_is_numerically_stable_for_large_logits():
    x = torch.tensor([[1000.0, 1001.0, 1002.0]])
    out = softmax(x, dim=-1)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.sum(dim=-1), torch.ones(1), atol=1e-6)


def test_softmax_rows_sum_to_one_along_chosen_dim():
    x = torch.randn(3, 5, 7)
    assert torch.allclose(softmax(x, dim=1).sum(dim=1), torch.ones(3, 7), atol=1e-6)


def test_attention_output_shape():
    q, k, v = (torch.randn(2, 3, 6, 8) for _ in range(3))
    assert scaled_dot_product_attention(q, k, v).shape == (2, 3, 6, 8)


def test_attention_averages_values_when_scores_are_uniform():
    """Zero queries => uniform attention => output is the mean of the value vectors."""
    q = torch.zeros(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    out = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, v.mean(dim=1, keepdim=True).expand_as(out), atol=1e-6)


def test_attention_applies_the_scaling_factor():
    q, k, v = torch.randn(1, 3, 16), torch.randn(1, 3, 16), torch.randn(1, 3, 16)
    scores = q @ k.transpose(-2, -1) / math.sqrt(16)
    expected = torch.softmax(scores, dim=-1) @ v
    assert torch.allclose(scaled_dot_product_attention(q, k, v), expected, atol=1e-6)


def test_mask_blocks_attention_to_masked_positions():
    q, k = torch.randn(1, 2, 4), torch.randn(1, 2, 4)
    v = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 99.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, False], [True, False]])  # position 1 is never attended
    out = scaled_dot_product_attention(q, k, v, mask=mask)
    assert torch.allclose(out, v[:, :1].expand_as(out), atol=1e-6)


def test_multihead_output_shape():
    attn = CausalMultiHeadSelfAttention(d_model=32, num_heads=4, max_seq_len=16, theta=10_000.0)
    assert attn(torch.randn(2, 6, 32)).shape == (2, 6, 32)


def test_multihead_is_causal():
    """Changing a future token must not change the output at an earlier position."""
    torch.manual_seed(0)
    attn = CausalMultiHeadSelfAttention(d_model=16, num_heads=2, max_seq_len=16, theta=10_000.0)
    x = torch.randn(1, 6, 16)
    baseline = attn(x)

    perturbed = x.clone()
    perturbed[:, 4:] = torch.randn(1, 2, 16)
    assert torch.allclose(attn(perturbed)[:, :4], baseline[:, :4], atol=1e-5)
    assert not torch.allclose(attn(perturbed)[:, 4:], baseline[:, 4:], atol=1e-5)


def test_rejects_indivisible_head_split():
    try:
        CausalMultiHeadSelfAttention(d_model=10, num_heads=4)
    except ValueError:
        return
    raise AssertionError("expected ValueError for d_model not divisible by num_heads")
