import pickle

import pytest

from tfs.tokenizer.bpe import Tokenizer, train_bpe

# Classic toy corpus from Sennrich et al. (2016), "Neural Machine Translation of Rare
# Words with Subword Units" -- used here only as a well-known, deterministic input,
# not copied from any course material.
TOY_CORPUS = ("low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3).strip()


@pytest.fixture()
def toy_corpus_path(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text(TOY_CORPUS, encoding="utf-8")
    return str(path)


def test_train_bpe_respects_vocab_size(toy_corpus_path):
    vocab, merges = train_bpe(toy_corpus_path, vocab_size=262)
    assert len(vocab) == 262
    assert len(merges) == 262 - 256


def test_train_bpe_merges_are_valid_and_ordered(toy_corpus_path):
    vocab, merges = train_bpe(toy_corpus_path, vocab_size=270)
    byte_values = set(vocab.values())
    for a, b in merges:
        assert a + b in byte_values


def test_train_bpe_stops_early_if_out_of_pairs(toy_corpus_path):
    # Asking for a huge vocab on a tiny corpus should stop once no pair repeats,
    # not crash or hang.
    vocab, merges = train_bpe(toy_corpus_path, vocab_size=100_000)
    assert len(vocab) == 256 + len(merges)
    assert len(vocab) < 100_000


def test_special_tokens_are_not_merged_across(tmp_path):
    text = "low low<|endoftext|>widest widest"
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")

    vocab, merges = train_bpe(str(path), vocab_size=260, special_tokens=["<|endoftext|>"])
    special_bytes = "<|endoftext|>".encode("utf-8")
    assert special_bytes in vocab.values()
    # The special token must never appear as one side of a learned merge.
    for a, b in merges:
        assert a != special_bytes and b != special_bytes


def test_encode_decode_roundtrip(toy_corpus_path):
    vocab, merges = train_bpe(toy_corpus_path, vocab_size=270)
    tok = Tokenizer(vocab, merges)
    text = "the lowest newest widest"
    assert tok.decode(tok.encode(text)) == text


def test_encode_decode_roundtrip_unicode(toy_corpus_path):
    vocab, merges = train_bpe(toy_corpus_path, vocab_size=270)
    tok = Tokenizer(vocab, merges)
    text = "héllo wörld 😀 — widest"
    assert tok.decode(tok.encode(text)) == text


def test_encode_compresses_frequent_pretokens(toy_corpus_path):
    vocab, merges = train_bpe(toy_corpus_path, vocab_size=270)
    tok = Tokenizer(vocab, merges)
    ids = tok.encode("newest")
    assert len(ids) < len("newest".encode("utf-8"))


def test_special_token_encodes_as_single_id(toy_corpus_path):
    vocab, merges = train_bpe(toy_corpus_path, vocab_size=270, special_tokens=["<|endoftext|>"])
    tok = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    ids = tok.encode("low<|endoftext|>widest")
    special_id = tok.byte_to_id["<|endoftext|>".encode("utf-8")]
    assert special_id in ids
    assert ids.count(special_id) == 1


def test_encode_iterable_matches_encode(toy_corpus_path):
    # Splitting exactly on a newline is a safe pretoken boundary (the GPT-2 regex always
    # closes out a whitespace pretoken at "\n"); splitting mid-word or mid-space-run is not,
    # since a leading space can belong to the *next* pretoken.
    vocab, merges = train_bpe(toy_corpus_path, vocab_size=270)
    tok = Tokenizer(vocab, merges)
    lines = ["the lowest\n", "newest widest"]
    assert list(tok.encode_iterable(lines)) == tok.encode("".join(lines))


def test_save_and_load_roundtrip(toy_corpus_path, tmp_path):
    vocab, merges = train_bpe(toy_corpus_path, vocab_size=270, special_tokens=["<|endoftext|>"])
    tok = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    vocab_path = tmp_path / "vocab.pkl"
    merges_path = tmp_path / "merges.pkl"
    tok.save(str(vocab_path), str(merges_path))

    loaded = Tokenizer.from_files(str(vocab_path), str(merges_path), special_tokens=["<|endoftext|>"])
    text = "low<|endoftext|>newest widest"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(loaded.encode(text)) == text
