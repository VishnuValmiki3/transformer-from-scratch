import math

import torch
import torch.nn.functional as F

from tfs.nn import cross_entropy, perplexity


def test_matches_torch_cross_entropy():
    logits = torch.randn(8, 20)
    targets = torch.randint(0, 20, (8,))
    assert torch.allclose(cross_entropy(logits, targets), F.cross_entropy(logits, targets), atol=1e-6)


def test_matches_torch_on_sequence_shaped_logits():
    logits = torch.randn(4, 7, 30)
    targets = torch.randint(0, 30, (4, 7))
    expected = F.cross_entropy(logits.reshape(-1, 30), targets.reshape(-1))
    assert torch.allclose(cross_entropy(logits, targets), expected, atol=1e-6)


def test_is_stable_for_huge_logits():
    """exp() of these raw logits would overflow to inf; the shift must prevent that."""
    logits = torch.tensor([[0.0, 1e4, 2e4]])
    targets = torch.tensor([2])
    loss = cross_entropy(logits, targets)
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_is_stable_for_very_negative_logits():
    logits = torch.tensor([[-1e4, -1e4, -1e4]])
    loss = cross_entropy(logits, torch.tensor([0]))
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(math.log(3)), atol=1e-5)


def test_uniform_logits_give_log_vocab_size():
    vocab_size = 50
    logits = torch.zeros(3, vocab_size)
    targets = torch.randint(0, vocab_size, (3,))
    assert torch.allclose(
        cross_entropy(logits, targets), torch.tensor(math.log(vocab_size)), atol=1e-6
    )


def test_confident_correct_prediction_gives_near_zero_loss():
    logits = torch.tensor([[20.0, 0.0, 0.0]])
    assert cross_entropy(logits, torch.tensor([0])).item() < 1e-6


def test_gradients_flow_through_loss():
    logits = torch.randn(4, 10, requires_grad=True)
    cross_entropy(logits, torch.randint(0, 10, (4,))).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_perplexity_is_exp_of_loss():
    assert math.isclose(perplexity(math.log(7.0)), 7.0, rel_tol=1e-5)
    assert math.isclose(perplexity(torch.tensor(0.0)), 1.0, rel_tol=1e-6)
