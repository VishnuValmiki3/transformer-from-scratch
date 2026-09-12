import pytest
import torch

from tfs.nn import RotaryPositionalEmbedding


def test_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        RotaryPositionalEmbedding(10_000.0, d_k=7, max_seq_len=16)


def test_preserves_shape():
    rope = RotaryPositionalEmbedding(10_000.0, d_k=8, max_seq_len=32)
    x = torch.randn(2, 4, 6, 8)
    assert rope(x, torch.arange(6)).shape == x.shape


def test_position_zero_is_identity():
    rope = RotaryPositionalEmbedding(10_000.0, d_k=8, max_seq_len=32)
    x = torch.randn(1, 8)
    assert torch.allclose(rope(x, torch.tensor([0])), x, atol=1e-6)


def test_rotation_preserves_norm():
    """RoPE applies a rotation, so vector magnitude must be unchanged."""
    rope = RotaryPositionalEmbedding(10_000.0, d_k=16, max_seq_len=64)
    x = torch.randn(3, 5, 16)
    rotated = rope(x, torch.arange(5))
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_dot_product_depends_only_on_relative_position():
    """The reason RoPE works: <R_i q, R_j k> is a function of (i - j) alone."""
    rope = RotaryPositionalEmbedding(10_000.0, d_k=16, max_seq_len=128)
    q, k = torch.randn(1, 16), torch.randn(1, 16)

    def dot_at(i, j):
        qi = rope(q, torch.tensor([i]))
        kj = rope(k, torch.tensor([j]))
        return (qi * kj).sum()

    assert torch.allclose(dot_at(2, 5), dot_at(2 + 30, 5 + 30), atol=1e-4)
    assert not torch.allclose(dot_at(2, 5), dot_at(2, 9), atol=1e-3)


def test_different_positions_give_different_encodings():
    rope = RotaryPositionalEmbedding(10_000.0, d_k=8, max_seq_len=32)
    x = torch.randn(1, 8)
    assert not torch.allclose(rope(x, torch.tensor([3])), rope(x, torch.tensor([4])), atol=1e-3)


def test_cache_is_not_persisted_in_state_dict():
    """cos/sin caches are derived from config, so they must not bloat checkpoints."""
    rope = RotaryPositionalEmbedding(10_000.0, d_k=8, max_seq_len=32)
    assert rope.state_dict() == {}
