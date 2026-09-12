import numpy as np
import pytest
import torch

from tfs.data.dataset import TOKEN_DTYPE, get_batch, load_tokens


@pytest.fixture()
def tokens():
    return np.arange(500, dtype=TOKEN_DTYPE)


def test_batch_shapes(tokens):
    inputs, targets = get_batch(tokens, batch_size=8, context_length=16)
    assert inputs.shape == (8, 16)
    assert targets.shape == (8, 16)


def test_targets_are_inputs_shifted_by_one(tokens):
    inputs, targets = get_batch(tokens, batch_size=4, context_length=10)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_windows_are_contiguous_spans_of_the_corpus(tokens):
    """With an arange corpus, every sampled window must be consecutive integers."""
    inputs, _ = get_batch(tokens, batch_size=6, context_length=12)
    steps = inputs[:, 1:] - inputs[:, :-1]
    assert torch.all(steps == 1)


def test_is_reproducible_with_a_seeded_generator(tokens):
    first, _ = get_batch(tokens, 4, 8, generator=np.random.default_rng(42))
    second, _ = get_batch(tokens, 4, 8, generator=np.random.default_rng(42))
    assert torch.equal(first, second)


def test_different_seeds_sample_different_windows(tokens):
    first, _ = get_batch(tokens, 8, 8, generator=np.random.default_rng(1))
    second, _ = get_batch(tokens, 8, 8, generator=np.random.default_rng(2))
    assert not torch.equal(first, second)


def test_emits_int64_for_embedding_lookup(tokens):
    inputs, targets = get_batch(tokens, 2, 4)
    assert inputs.dtype == torch.int64
    assert targets.dtype == torch.int64


def test_never_reads_past_the_end_of_the_corpus():
    short = np.arange(21, dtype=TOKEN_DTYPE)
    inputs, targets = get_batch(short, batch_size=64, context_length=20)
    assert targets.max().item() <= 20


def test_rejects_corpus_shorter_than_context():
    with pytest.raises(ValueError):
        get_batch(np.arange(5, dtype=TOKEN_DTYPE), batch_size=1, context_length=10)


def test_load_tokens_reads_a_bin_file_without_copying(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(100, dtype=TOKEN_DTYPE).tofile(path)

    loaded = load_tokens(str(path))
    assert isinstance(loaded, np.memmap)
    assert len(loaded) == 100
    assert loaded[99] == 99
