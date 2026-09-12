"""Turn a raw text corpus into a trained BPE tokenizer plus uint16 .bin token files.

Two deliberate engineering choices, both worth knowing about:

1. BPE merges are learned from a bounded sample of the training text, not all of it.
   Merge statistics saturate quickly, and this keeps a pure-Python trainer tractable.
2. The corpus is split into documents on the special token before encoding, which is the
   only split point guaranteed to tokenize identically to the un-split corpus (the special
   token is always atomic, so no pretoken can straddle the boundary).
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import time
import urllib.request

import numpy as np

from ..tokenizer.bpe import Tokenizer, train_bpe
from .dataset import TOKEN_DTYPE

# Public mirrors of the TinyStories corpus. Override with --train-url/--valid-url if these move.
TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

END_OF_TEXT = "<|endoftext|>"

_WORKER_TOKENIZER: Tokenizer | None = None


def download(url: str, dest: str) -> str:
    if os.path.exists(dest):
        print(f"already present: {dest}")
        return dest
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    print(f"downloading {url}\n        -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def write_prefix(source: str, dest: str, max_chars: int) -> str:
    """Copy a prefix of the corpus, trimmed to the last complete document."""
    with open(source, "r", encoding="utf-8") as f:
        text = f.read(max_chars)
    cutoff = text.rfind(END_OF_TEXT)
    if cutoff > 0:
        text = text[: cutoff + len(END_OF_TEXT)]
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    return dest


def iter_documents(path: str, block_size: int = 1 << 22):
    """Stream the corpus as documents delimited by the special token."""
    buffer = ""
    with open(path, "r", encoding="utf-8") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            buffer += block
            parts = buffer.split(END_OF_TEXT)
            buffer = parts.pop()
            for part in parts:
                yield part + END_OF_TEXT
    if buffer.strip():
        yield buffer


def _init_worker(vocab, merges, special_tokens) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer(vocab, merges, special_tokens)


def _encode_document(document: str) -> list[int]:
    assert _WORKER_TOKENIZER is not None
    return _WORKER_TOKENIZER.encode(document)


def encode_corpus(
    tokenizer: Tokenizer,
    source: str,
    dest: str,
    workers: int = 1,
) -> int:
    start = time.perf_counter()
    total_tokens = 0
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)

    with open(dest, "wb") as out:
        if workers <= 1:
            for document in iter_documents(source):
                ids = tokenizer.encode(document)
                np.asarray(ids, dtype=TOKEN_DTYPE).tofile(out)
                total_tokens += len(ids)
        else:
            merges = list(tokenizer.merge_ranks.keys())
            with multiprocessing.Pool(
                workers,
                initializer=_init_worker,
                initargs=(tokenizer.vocab, merges, tokenizer.special_tokens),
            ) as pool:
                for ids in pool.imap(_encode_document, iter_documents(source), chunksize=64):
                    np.asarray(ids, dtype=TOKEN_DTYPE).tofile(out)
                    total_tokens += len(ids)

    elapsed = time.perf_counter() - start
    source_bytes = os.path.getsize(source)
    print(
        f"encoded {source} -> {dest}\n"
        f"  {total_tokens:,} tokens in {elapsed:.1f}s "
        f"({total_tokens / max(elapsed, 1e-9):,.0f} tok/s, "
        f"{source_bytes / max(total_tokens, 1):.2f} bytes/token)"
    )
    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/tinystories")
    parser.add_argument("--train-file", help="local training text (skips download)")
    parser.add_argument("--valid-file", help="local validation text (skips download)")
    parser.add_argument("--train-url", default=TRAIN_URL)
    parser.add_argument("--valid-url", default=VALID_URL)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument(
        "--bpe-sample-chars",
        type=int,
        default=50_000_000,
        help="how much of the training text to learn merges from",
    )
    parser.add_argument(
        "--max-train-chars",
        type=int,
        help="encode only this much of the training text; a 10k-step run consumes far "
        "fewer tokens than the full corpus holds, so this saves most of the prep time",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.vocab_size > np.iinfo(TOKEN_DTYPE).max + 1:
        raise ValueError(f"vocab_size {args.vocab_size} exceeds what {TOKEN_DTYPE.__name__} can hold")

    os.makedirs(args.data_dir, exist_ok=True)
    train_text = args.train_file or download(
        args.train_url, os.path.join(args.data_dir, "train.txt")
    )
    valid_text = args.valid_file or download(
        args.valid_url, os.path.join(args.data_dir, "valid.txt")
    )

    sample = write_prefix(
        train_text, os.path.join(args.data_dir, "bpe_sample.txt"), args.bpe_sample_chars
    )
    if args.max_train_chars:
        train_text = write_prefix(
            train_text, os.path.join(args.data_dir, "train_subset.txt"), args.max_train_chars
        )
        print(f"encoding a {os.path.getsize(train_text):,}-byte subset of the training text")

    print(f"training BPE (vocab_size={args.vocab_size}) on {os.path.getsize(sample):,} bytes")
    start = time.perf_counter()
    vocab, merges = train_bpe(sample, args.vocab_size, special_tokens=[END_OF_TEXT])
    print(f"  learned {len(merges):,} merges in {time.perf_counter() - start:.1f}s")
    print(f"  actual vocab size: {len(vocab):,}")
    if len(vocab) < args.vocab_size:
        print(
            f"  WARNING: the sample ran out of repeated pairs before reaching "
            f"{args.vocab_size:,}. Set vocab_size: {len(vocab)} in your training config, "
            f"or the model will have {args.vocab_size - len(vocab):,} output slots that "
            f"decode to nothing."
        )

    tokenizer = Tokenizer(vocab, merges, special_tokens=[END_OF_TEXT])
    tokenizer.save(
        os.path.join(args.data_dir, "vocab.pkl"), os.path.join(args.data_dir, "merges.pkl")
    )

    encode_corpus(tokenizer, train_text, os.path.join(args.data_dir, "train.bin"), args.workers)
    encode_corpus(tokenizer, valid_text, os.path.join(args.data_dir, "valid.bin"), args.workers)


if __name__ == "__main__":
    main()
