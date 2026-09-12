"""Encode a style-transfer corpus with the *pretraining* tokenizer.

Reusing the base model's vocab is mandatory, not a convenience: LoRA adapts a model whose
embedding table is indexed by those exact token ids, so a freshly trained tokenizer would
silently scramble every input.
"""

from __future__ import annotations

import argparse
import os
import urllib.request

import numpy as np

from ..tokenizer.bpe import Tokenizer
from .dataset import TOKEN_DTYPE
from .prepare_tinystories import END_OF_TEXT

# Karpathy's tiny-shakespeare, the usual small style-transfer corpus.
DEFAULT_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/style")
    parser.add_argument("--input-file", help="local text file (skips download)")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--vocab", required=True, help="vocab.pkl from pretraining")
    parser.add_argument("--merges", required=True, help="merges.pkl from pretraining")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    source = args.input_file
    if source is None:
        source = os.path.join(args.data_dir, "corpus.txt")
        if not os.path.exists(source):
            print(f"downloading {args.url}")
            urllib.request.urlretrieve(args.url, source)

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=[END_OF_TEXT])
    with open(source, "r", encoding="utf-8") as f:
        text = f.read()

    ids = np.asarray(tokenizer.encode(text), dtype=TOKEN_DTYPE)
    split = int(len(ids) * (1 - args.val_fraction))

    train_path = os.path.join(args.data_dir, "train.bin")
    val_path = os.path.join(args.data_dir, "valid.bin")
    ids[:split].tofile(train_path)
    ids[split:].tofile(val_path)

    print(
        f"{source}: {len(text):,} chars -> {len(ids):,} tokens "
        f"({len(text) / max(len(ids), 1):.2f} bytes/token)\n"
        f"  train {split:,} -> {train_path}\n"
        f"  val   {len(ids) - split:,} -> {val_path}"
    )


if __name__ == "__main__":
    main()
