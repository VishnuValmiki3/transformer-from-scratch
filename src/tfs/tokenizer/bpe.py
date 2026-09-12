"""Byte-level BPE tokenizer, trained and applied from scratch (no tokenizers/tiktoken dependency).

Pretokenization uses the GPT-2 regex pattern; training merges the most frequent adjacent
byte-pair at each step, breaking ties by preferring the lexicographically greater pair.
"""

from __future__ import annotations

import pickle
from collections import Counter, defaultdict

import regex as re

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

Pair = tuple[bytes, bytes]
REPLACEMENT_CHAR = "�".encode("utf-8")


def _iter_pretokens(text: str):
    for match in re.finditer(GPT2_PAT, text):
        yield match.group()


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    # Longest-first so overlapping special tokens (one a prefix of another) match correctly.
    ordered = sorted(special_tokens, key=len, reverse=True)
    pattern = "(" + "|".join(re.escape(tok) for tok in ordered) + ")"
    return re.split(pattern, text)


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    """Count pretoken byte-sequences, excluding special-token spans (they never get merged)."""
    counts: Counter[tuple[bytes, ...]] = Counter()
    special_set = set(special_tokens)
    for chunk in _split_on_special_tokens(text, special_tokens):
        if not chunk or chunk in special_set:
            continue
        for pretoken in _iter_pretokens(chunk):
            key = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
            counts[key] += 1
    return counts


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[Pair]]:
    """Train a byte-level BPE tokenizer on the text file at `input_path`.

    Returns (vocab, merges): vocab maps token id -> raw bytes, merges is the ordered
    list of byte-pairs learned (earlier merges applied first during encoding).
    """
    special_tokens = special_tokens or []

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    if vocab_size < len(vocab):
        raise ValueError(f"vocab_size ({vocab_size}) must be >= base vocab size ({len(vocab)})")
    num_merges = vocab_size - len(vocab)

    pretoken_counts = _pretoken_counts(text, special_tokens)
    words: list[list[bytes]] = [list(key) for key in pretoken_counts]
    freqs: list[int] = list(pretoken_counts.values())

    pair_counts: Counter[Pair] = Counter()
    pair_index: dict[Pair, set[int]] = defaultdict(set)

    def add_word(idx: int) -> None:
        w, freq = words[idx], freqs[idx]
        for pair in zip(w, w[1:]):
            pair_counts[pair] += freq
            pair_index[pair].add(idx)

    def remove_word(idx: int) -> None:
        w, freq = words[idx], freqs[idx]
        for pair in set(zip(w, w[1:])):
            occurrences = pair_index.get(pair)
            if occurrences is not None:
                occurrences.discard(idx)
                if not occurrences:
                    del pair_index[pair]
        for pair in zip(w, w[1:]):
            pair_counts[pair] -= freq
            if pair_counts[pair] <= 0:
                pair_counts.pop(pair, None)

    for idx in range(len(words)):
        add_word(idx)

    merges: list[Pair] = []
    for _ in range(num_merges):
        if not pair_counts:
            break
        best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        affected = list(pair_index.get(best_pair, ()))
        for idx in affected:
            remove_word(idx)
        for idx in affected:
            w = words[idx]
            merged, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best_pair:
                    merged.append(w[i] + w[i + 1])
                    i += 2
                else:
                    merged.append(w[i])
                    i += 1
            words[idx] = merged
            add_word(idx)
        pair_counts.pop(best_pair, None)
        pair_index.pop(best_pair, None)

    return vocab, merges


class Tokenizer:
    """Encode/decode text using a trained (vocab, merges) pair, with optional special tokens."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.special_tokens = special_tokens or []
        self.byte_to_id = {b: i for i, b in self.vocab.items()}

        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in self.byte_to_id:
                new_id = max(self.vocab) + 1
                self.vocab[new_id] = b
                self.byte_to_id[b] = new_id

        self.merge_ranks: dict[Pair, int] = {pair: rank for rank, pair in enumerate(merges)}
        # Pretokens repeat heavily in natural text, so memoizing the merge search per
        # distinct pretoken is what makes encoding a multi-GB corpus tractable in Python.
        self._cache: dict[bytes, list[int]] = {}

    @classmethod
    def from_files(
        cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def save(self, vocab_path: str, merges_path: str) -> None:
        with open(vocab_path, "wb") as f:
            pickle.dump(self.vocab, f)
        with open(merges_path, "wb") as f:
            pickle.dump(list(self.merge_ranks.keys()), f)

    def _encode_pretoken(self, raw: bytes) -> list[bytes]:
        symbols = [bytes([b]) for b in raw]
        while len(symbols) > 1:
            pairs = list(zip(symbols, symbols[1:]))
            ranked = [
                (self.merge_ranks[p], i) for i, p in enumerate(pairs) if p in self.merge_ranks
            ]
            if not ranked:
                break
            _, i = min(ranked)
            symbols = symbols[:i] + [symbols[i] + symbols[i + 1]] + symbols[i + 2 :]
        return symbols

    def _encode_pretoken_ids(self, raw: bytes) -> list[int]:
        cached = self._cache.get(raw)
        if cached is None:
            cached = [self.byte_to_id[s] for s in self._encode_pretoken(raw)]
            self._cache[raw] = cached
        return cached

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        special_set = set(self.special_tokens)
        for chunk in _split_on_special_tokens(text, self.special_tokens):
            if chunk == "":
                continue
            if chunk in special_set:
                ids.append(self.byte_to_id[chunk.encode("utf-8")])
                continue
            for pretoken in _iter_pretokens(chunk):
                ids.extend(self._encode_pretoken_ids(pretoken.encode("utf-8")))
        return ids

    def encode_iterable(self, iterable):
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        # A model's vocab is often padded past the tokenizer's (for alignment, or because
        # training ran out of merges), so sampling can legitimately produce unknown ids.
        # Render them like any other undecodable byte rather than raising.
        raw = b"".join(self.vocab.get(i, REPLACEMENT_CHAR) for i in ids)
        return raw.decode("utf-8", errors="replace")
