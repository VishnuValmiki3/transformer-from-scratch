import math

import pytest
import torch

from tfs.nn import TransformerBlock, TransformerLM, cross_entropy

MODEL_KWARGS = dict(
    vocab_size=97,
    d_model=32,
    num_heads=4,
    d_ff=64,
    num_layers=2,
    max_seq_len=16,
    theta=10_000.0,
)


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return TransformerLM(**MODEL_KWARGS)


def test_block_preserves_shape():
    block = TransformerBlock(d_model=16, num_heads=2, d_ff=32, max_seq_len=8, theta=10_000.0)
    assert block(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_block_is_residual():
    """With both sublayers zeroed out, a pre-norm block must be the identity."""
    block = TransformerBlock(d_model=16, num_heads=2, d_ff=32, max_seq_len=8, theta=10_000.0)
    with torch.no_grad():
        block.attn.output_proj.weight.zero_()
        block.ffn.w2.weight.zero_()
    x = torch.randn(2, 5, 16)
    assert torch.allclose(block(x), x, atol=1e-6)


def test_forward_logits_shape(model):
    logits = model(torch.randint(0, 97, (3, 12)))
    assert logits.shape == (3, 12, 97)


def test_forward_is_causal(model):
    """End-to-end causality: editing the last token can't move earlier-position logits."""
    tokens = torch.randint(0, 97, (1, 10))
    baseline = model(tokens)

    edited = tokens.clone()
    edited[0, -1] = (edited[0, -1] + 1) % 97
    assert torch.allclose(model(edited)[:, :-1], baseline[:, :-1], atol=1e-5)


def test_rejects_sequence_longer_than_context(model):
    with pytest.raises(ValueError):
        model(torch.randint(0, 97, (1, MODEL_KWARGS["max_seq_len"] + 1)))


def test_embedding_and_output_head_are_tied(model):
    assert model.lm_head.weight is model.token_embedding.weight


def test_untied_model_has_independent_output_head():
    untied = TransformerLM(**MODEL_KWARGS, tie_embeddings=False)
    assert untied.lm_head.weight is not untied.token_embedding.weight
    assert untied.num_parameters() > TransformerLM(**MODEL_KWARGS).num_parameters()


def test_tying_saves_exactly_one_embedding_matrix(model):
    """Tying must remove the output head's copy of the vocab-sized matrix, and PyTorch's
    parameters() must yield the shared tensor once so an optimizer can't step it twice."""
    untied = TransformerLM(**MODEL_KWARGS, tie_embeddings=False)
    embedding_size = model.token_embedding.weight.numel()
    assert untied.num_parameters() - model.num_parameters() == embedding_size
    assert len(list(model.parameters())) == len(list(untied.parameters())) - 1


@pytest.mark.parametrize("tie_embeddings", [True, False])
def test_initial_loss_is_near_uniform_prediction(tie_embeddings):
    """A fresh model should predict roughly uniformly, i.e. loss ~ ln(vocab_size).

    Regression test for an init/tying interaction: with a std=1 embedding, tying made the
    output logits scale with sqrt(d_model), starting training at loss ~342 instead of ~9.2.
    """
    vocab_size = 10_000
    torch.manual_seed(0)
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=1344,
        num_layers=2,
        max_seq_len=64,
        tie_embeddings=tie_embeddings,
    )
    tokens = torch.randint(0, vocab_size, (2, 32))
    targets = torch.randint(0, vocab_size, (2, 32))

    with torch.no_grad():
        loss = cross_entropy(model(tokens), targets).item()
    assert abs(loss - math.log(vocab_size)) < 0.5, f"initial loss {loss:.2f} is far from uniform"


def test_gradients_reach_every_parameter(model):
    logits = model(torch.randint(0, 97, (2, 8)))
    logits.sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"
