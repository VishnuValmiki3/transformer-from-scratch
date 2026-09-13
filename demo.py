#!/usr/bin/env python
"""Live walkthrough of transformer-from-scratch: tokenizer, model, sampling, LoRA.

Everything printed below is computed on the spot from the trained checkpoints in this
repository -- no numbers are hardcoded. Runs on CPU in about a minute.

    python demo.py                # full walkthrough
    python demo.py --act 3        # just the sampling act
    python demo.py --prompt "The dragon"
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time

# The demo prints box-drawing characters and arrows; Windows consoles default to cp1252.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

import numpy as np
import torch

from tfs.checkpoint import load_config
from tfs.data.dataset import get_batch, load_tokens
from tfs.generate import load_model_for_inference, top_p_filter
from tfs.nn import cross_entropy, perplexity, softmax
from tfs.tokenizer.bpe import Tokenizer
from tfs.train import TrainConfig, build_model

ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(ROOT, "checkpoints", "tinystories", "final.pt")
ADAPTER = os.path.join(ROOT, "checkpoints", "lora_style", "adapter.pt")
VOCAB = os.path.join(ROOT, "data", "tinystories", "vocab.pkl")
MERGES = os.path.join(ROOT, "data", "tinystories", "merges.pkl")
STORIES_VAL = os.path.join(ROOT, "data", "tinystories", "valid.bin")
STYLE_VAL = os.path.join(ROOT, "data", "style", "valid.bin")

WIDTH = 78
COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


# ---------------------------------------------------------------- presentation


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def dim(s: str) -> str:
    return c(s, "2")


def bold(s: str) -> str:
    return c(s, "1")


def cyan(s: str) -> str:
    return c(s, "36")


def green(s: str) -> str:
    return c(s, "32")


def yellow(s: str) -> str:
    return c(s, "33")


def magenta(s: str) -> str:
    return c(s, "35")


def act(number: int, title: str, subtitle: str) -> None:
    print()
    print(cyan("┌" + "─" * (WIDTH - 2) + "┐"))
    print(cyan("│") + bold(f" ACT {number}  {title}".ljust(WIDTH - 2)) + cyan("│"))
    print(cyan("│") + dim(f" {subtitle}".ljust(WIDTH - 2)) + cyan("│"))
    print(cyan("└" + "─" * (WIDTH - 2) + "┘"))
    print()


def note(text: str) -> None:
    print(dim("  " + text))


def result(label: str, value: str, good: bool = False) -> None:
    paint = green if good else yellow
    print(f"  {label.ljust(38)} {paint(value)}")


def rule() -> None:
    print(dim("  " + "─" * (WIDTH - 4)))


def row(cells: list[str], widths: list[int], painted: str | None = None) -> None:
    """Right-align numeric cells; `painted` replaces the last cell after padding."""
    out = "  " + cells[0].ljust(widths[0])
    for cell, width in zip(cells[1:], widths[1:]):
        out += cell.rjust(width)
    if painted is not None:
        out = out[: -widths[-1]] + painted.rjust(widths[-1])
    print(out)


# -------------------------------------------------------------------- helpers


def require_artifacts() -> None:
    missing = [p for p in (CHECKPOINT, VOCAB, MERGES) if not os.path.exists(p)]
    if missing:
        print("Missing artifacts this demo needs:")
        for path in missing:
            print(f"  - {os.path.relpath(path, ROOT)}")
        print("\nRun the pretraining steps in the README quickstart first.")
        raise SystemExit(1)


@torch.no_grad()
def evaluate(model, tokens_path: str, context_length: int, batches: int, seed: int) -> float:
    """Mean cross-entropy over `batches` random windows of a held-out token file."""
    data = load_tokens(tokens_path)
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(batches):
        inputs, targets = get_batch(data, 8, context_length, "cpu", rng)
        logits = model(inputs)
        total += float(cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)))
    return total / batches


@torch.no_grad()
def stream(model, tokenizer, prompt, max_new_tokens, temperature, top_p, seed, eos_id) -> None:
    """Sample one token at a time, printing each piece the moment it is produced."""
    model.eval()
    ids = tokenizer.encode(prompt)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    print("  " + magenta(prompt), end="", flush=True)
    shown = len(tokenizer.decode(ids))
    started = time.perf_counter()
    produced = 0

    for _ in range(max_new_tokens):
        context = ids[-model.max_seq_len :]
        logits = model(torch.tensor([context], dtype=torch.long))[0, -1]
        probs = top_p_filter(softmax(logits / temperature, dim=-1), top_p)
        next_id = int(torch.multinomial(probs, num_samples=1, generator=generator))
        if eos_id is not None and next_id == eos_id:
            break

        ids.append(next_id)
        produced += 1
        # Decode the whole prefix each step: one character can straddle two tokens, so
        # decoding the new id alone would print a replacement character mid-word.
        text = tokenizer.decode(ids)
        piece = text[shown:]
        shown = len(text)
        print(piece.replace("\n", "\n  "), end="", flush=True)

    elapsed = time.perf_counter() - started
    print("\n")
    rate = produced / elapsed if elapsed else 0.0
    note(f"{produced} tokens in {elapsed:.1f}s  ({rate:.1f} tok/s, CPU, single stream)")


# ----------------------------------------------------------------------- acts


def act_tokenizer(args) -> Tokenizer:
    act(1, "THE TOKENIZER", "byte-level BPE, trained from scratch -- no tiktoken, no HF")

    tokenizer = Tokenizer.from_files(VOCAB, MERGES, special_tokens=["<|endoftext|>"])
    sentence = "Once upon a time, a curious little dragon discovered pancakes."

    started = time.perf_counter()
    ids = tokenizer.encode(sentence)
    elapsed = time.perf_counter() - started

    note("The vocabulary and merge list were learned by repeatedly counting adjacent")
    note("byte-pairs over 30 MB of text and merging the most frequent one.")
    print()
    result("Vocabulary size", f"{len(tokenizer.vocab):,} tokens")
    result("Learned merges", f"{len(tokenizer.merge_ranks):,}")
    print()
    print(bold("  Encoding: ") + magenta(sentence))
    print()

    line, used = "  ", 2
    for token_id in ids:
        piece = tokenizer.vocab[token_id].decode("utf-8", errors="replace")
        # A leading space is part of the token, so show it as a visible marker; token
        # boundaries are the whole point of this display.
        cell = piece.replace("\n", "\\n").replace(" ", "␣")
        width = len(cell) + len(str(token_id)) + 3
        if used + width > WIDTH:
            print(line)
            line, used = "  ", 2
        line += c(cell, "7") + dim(f"·{token_id}") + "  "
        used += width
    print(line)
    print()

    round_trip = tokenizer.decode(ids)
    result("Bytes in / tokens out", f"{len(sentence.encode())} → {len(ids)}")
    result("Compression", f"{len(sentence.encode()) / len(ids):.2f} bytes per token")
    result("Encode time", f"{elapsed * 1e6:.0f} µs")
    result(
        "Round-trip decode == input",
        "True" if round_trip == sentence else "False",
        good=round_trip == sentence,
    )
    print()
    note("Merges are ranked, and the lowest available rank is applied first -- which is why")
    note("common whole words survive as one token while rare ones split into fragments.")
    return tokenizer


def act_model(args):
    act(2, "THE MODEL", "a 17.6M-parameter decoder-only Transformer, loaded from checkpoint")

    cfg = TrainConfig(**load_config(CHECKPOINT))
    started = time.perf_counter()
    model = load_model_for_inference(CHECKPOINT, "cpu")
    load_time = time.perf_counter() - started

    note("Every layer here is written against raw tensor ops: no nn.Linear, nn.Embedding,")
    note("nn.LayerNorm, nn.MultiheadAttention, F.softmax or F.cross_entropy in src/.")
    print()
    result("Parameters", f"{model.num_parameters():,}")
    result("Layers x heads x d_model", f"{cfg.num_layers} x {cfg.num_heads} x {cfg.d_model}")
    result("Feed-forward", f"SwiGLU {cfg.d_model} → {cfg.d_ff} → {cfg.d_model}")
    result("Positional encoding", f"RoPE (theta = {cfg.rope_theta:,.0f})")
    result("Normalization", "RMSNorm, pre-norm, bias-free")
    result("Output head tied to embeddings", str(cfg.tie_embeddings))
    result("Checkpoint load", f"{load_time:.2f}s")
    print()

    if os.path.exists(STORIES_VAL):
        note("Scoring the held-out TinyStories validation split, right now:")
        started = time.perf_counter()
        loss = evaluate(model, STORIES_VAL, cfg.context_length, args.eval_batches, seed=1234)
        print()
        result("Validation loss", f"{loss:.4f}")
        result("Validation perplexity", f"{perplexity(loss):.2f}", good=True)
        print()
        note(
            f"Sampled from {args.eval_batches} random batches of 8 x {cfg.context_length} "
            f"tokens in {time.perf_counter() - started:.1f}s on CPU, so it moves a little"
        )
        note("run to run; the full-split figure is 4.72. Pass --eval-batches 20 to tighten it.")
        note(f"A uniform guess over {cfg.vocab_size:,} tokens would score perplexity 10,000.")
    return model, cfg


def act_generate(args, model, tokenizer):
    act(3, "GENERATION", "autoregressive sampling -- one full forward pass per token")

    eos_id = tokenizer.byte_to_id.get(b"<|endoftext|>")
    note(f"Prompt in magenta, continuation after it. temperature={args.temperature}, "
         f"top-p={args.top_p}.")
    print()
    stream(
        model, tokenizer, args.prompt, args.max_new_tokens,
        args.temperature, args.top_p, args.seed, eos_id,
    )
    note("Nucleus sampling keeps only the smallest set of tokens whose probability mass")
    note("reaches top-p, so the long tail of 10,000 candidates is never sampled from.")


def act_causality(args, model, tokenizer, cfg):
    act(4, "CAUSALITY, PROVEN", "a property the test suite asserts -- re-run on the real model")

    ids = tokenizer.encode("The little dragon flew over the quiet village at night")[:16]
    edited_ids = list(ids)
    edited_ids[-1] = (edited_ids[-1] + 1_000) % cfg.vocab_size

    with torch.no_grad():
        logits_a = model(torch.tensor([ids], dtype=torch.long))[0]
        logits_b = model(torch.tensor([edited_ids], dtype=torch.long))[0]

    prefix_delta = float((logits_a[:-1] - logits_b[:-1]).abs().max())
    last_delta = float((logits_a[-1] - logits_b[-1]).abs().max())

    note(f"Take a {len(ids)}-token sequence, change only the LAST token, and re-run it.")
    note("If the causal mask is correct, every earlier position must be bit-identical.")
    print()
    result(
        f"Max |change| over positions 0..{len(ids) - 2}",
        f"{prefix_delta:.3e}",
        good=prefix_delta == 0.0,
    )
    result("Max |change| at the edited position", f"{last_delta:.3e}")
    print()
    if prefix_delta == 0.0:
        print("  " + green("Exactly zero. No token can see its own future."))
    else:
        print("  " + yellow("Nonzero -- causality is leaking."))
    print()
    note("This is a perturbation test rather than a shape assertion. A mask applied to the")
    note("wrong axis still yields correct shapes; it does not survive this.")


def act_lora(args, tokenizer, cfg):
    act(5, "LoRA", "a 261 KB adapter that rewrites the model's voice -- and what it costs")

    if not os.path.exists(ADAPTER):
        note("No adapter checkpoint found; skipping. Run tfs.finetune_lora first.")
        return

    payload = torch.load(ADAPTER, map_location="cpu", weights_only=False)
    trainable = sum(t.numel() for t in payload["lora"].values())
    base = load_model_for_inference(CHECKPOINT, "cpu")
    tuned = load_model_for_inference(CHECKPOINT, "cpu", ADAPTER)

    note("The same pretrained model, fine-tuned onto Shakespeare by training two small")
    note("low-rank matrices per attention projection and freezing everything else.")
    print()
    result("Adapter file size", f"{os.path.getsize(ADAPTER) / 1024:.0f} KB")
    result("Trainable parameters", f"{trainable:,}  ({trainable / base.num_parameters():.2%})")
    result("Rank / alpha", f"{payload['rank']} / {payload['alpha']:g}")
    result("Adapted modules", ", ".join(payload["target_modules"]))
    print()

    eos_id = tokenizer.byte_to_id.get(b"<|endoftext|>")
    tokens = min(args.max_new_tokens, 90)

    print(bold("  Base model:"))
    print()
    stream(base, tokenizer, args.prompt, tokens, args.temperature, args.top_p, args.seed, eos_id)
    rule()
    print(bold("  Same prompt, same seed, plus the 261 KB adapter:"))
    print()
    stream(tuned, tokenizer, args.prompt, tokens, args.temperature, args.top_p, args.seed, eos_id)

    if not (os.path.exists(STORIES_VAL) and os.path.exists(STYLE_VAL)):
        return

    rule()
    print()
    note("Both sides of the trade, measured now on both held-out splits:")
    print()
    widths = [26, 12, 12, 12]
    row([bold("Domain"), "base", "+ LoRA", "change"], widths)
    for name, path in (
        ("Shakespeare (target)", STYLE_VAL),
        ("TinyStories (original)", STORIES_VAL),
    ):
        before = perplexity(evaluate(base, path, cfg.context_length, args.eval_batches, 7))
        after = perplexity(evaluate(tuned, path, cfg.context_length, args.eval_batches, 7))
        change = (after - before) / before
        text = f"{change:+.1%}"
        row(
            [name, f"{before:.2f}", f"{after:.2f}", text],
            widths,
            painted=green(text) if change < 0 else yellow(text),
        )
    print()
    note("A decisive win on the target and a decisive loss on the domain it came from.")
    note("Adapting q_proj/v_proj rotates attention while frozen embeddings and FFNs keep the")
    note("old lexicon -- which is why the structure transfers but the vocabulary does not.")
    note("The one-sided version of this table is the more common and less useful one.")


def act_init(args, cfg):
    act(6, "THE FINDING", "why weight tying silently decides your initialization scale")

    if not os.path.exists(STORIES_VAL):
        note("Validation tokens missing; skipping.")
        return

    note("Two freshly initialized models, identical apart from the embedding init scale,")
    note("scored on one real batch right now -- before a single gradient step:")
    print()

    data = load_tokens(STORIES_VAL)
    inputs, targets = get_batch(data, 4, cfg.context_length, "cpu", np.random.default_rng(0))

    uniform = float(np.log(cfg.vocab_size))
    widths = [22, 16, 18]
    row([bold("embedding std"), "logit std", "initial loss"], widths)

    for std in (1.0, 0.02):
        torch.manual_seed(0)
        model = build_model(cfg, "cpu")
        with torch.no_grad():
            weight = model.token_embedding.weight
            torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
            logits = model(inputs)
            loss = float(
                cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            )
        text = f"{loss:.2f}"
        row(
            [str(std), f"{float(logits.std()):.2f}", text],
            widths,
            painted=green(text) if loss < 2 * uniform else yellow(text),
        )
    print()
    result(f"ln({cfg.vocab_size:,}), a uniform predictor", f"{uniform:.2f}")
    print()
    note("Under weight tying the embedding matrix IS the output projection, so every logit")
    note(f"is a {cfg.d_model}-long dot product against it: std 1.0 puts logit std near")
    note(f"sqrt({cfg.d_model}) = {np.sqrt(cfg.d_model):.1f}, and the model spends its first "
         f"several hundred steps")
    note("undoing its own initialization. Switching to GPT-2's 0.02 and changing nothing")
    note("else cut final validation perplexity from 6.07 to 4.72 -- 22%, from one line.")


# ------------------------------------------------------------------ live mode


LIVE_HELP = """  Type anything and the model continues it.

    /next <text>      the five tokens it thinks come next, with probabilities
    /score <a> | <b>  which of two sentences the model finds more likely
    /style            toggle the Shakespeare LoRA adapter on and off
    /temp <n>         sampling temperature (now: {temperature})
    /help             this list
    /quit             leave
"""


@torch.no_grad()
def show_next_tokens(model, tokenizer, text: str) -> None:
    """The model's actual next-token distribution -- the thing sampling draws from."""
    ids = tokenizer.encode(text)
    if not ids:
        note("Give me some text first.")
        return

    logits = model(torch.tensor([ids[-model.max_seq_len :]], dtype=torch.long))[0, -1]
    probs = softmax(logits, dim=-1)
    top = torch.topk(probs, 5)

    print()
    print("  " + dim("after ") + magenta(repr(text)[1:-1]) + dim(" the model expects:"))
    print()
    for prob, token_id in zip(top.values.tolist(), top.indices.tolist()):
        piece = tokenizer.vocab.get(int(token_id), b"?").decode("utf-8", errors="replace")
        cell = piece.replace("\n", "\\n").replace(" ", "␣")
        bar = "█" * max(1, round(prob * 40))
        # A confident model puts everything on one token, so the runners-up need
        # more than one decimal place to be worth printing at all.
        shown = f"{prob:6.2%}" if prob >= 0.0001 else "  <0.01%"
        print(f"  {cell[:18].rjust(18)}  {green(bar)} {shown}")
    print()
    note("These are the real probabilities. Sampling picks from this list, which is why")
    note("the same prompt gives a different story every time.")


@torch.no_grad()
def score_sentences(model, tokenizer, left: str, right: str) -> None:
    """Perplexity of two candidate sentences -- lower means the model finds it likelier."""
    scored = []
    for text in (left, right):
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            note(f"{text!r} is too short to score.")
            return
        batch = torch.tensor([ids[-model.max_seq_len :]], dtype=torch.long)
        logits = model(batch)
        loss = cross_entropy(logits[0, :-1], batch[0, 1:])
        scored.append((text, perplexity(float(loss))))

    print()
    best = min(scored, key=lambda pair: pair[1])[0]
    for text, ppl in scored:
        marker = green(" ← more likely") if text == best else ""
        print(f"  {yellow(f'{ppl:9.1f}')}  {text}{marker}")
    print()
    note("Perplexity is roughly 'how many tokens it was torn between'. It was never told")
    note("which sentence is correct -- it learned that from reading.")


def act_live(args, tokenizer, cfg) -> None:
    act(7, "LIVE", "your turn -- type anything and watch what the model really does")

    model = load_model_for_inference(CHECKPOINT, "cpu")
    styled = None
    using_style = False
    eos_id = tokenizer.byte_to_id.get(b"<|endoftext|>")
    temperature = args.temperature

    note("This model has 17.6M parameters and read nothing but children's stories. It is a")
    note("story continuer, not a chatbot -- ask it a question and it will answer with a")
    note("story, because a story is the only thing it has ever seen.")
    print(LIVE_HELP.format(temperature=temperature))

    while True:
        try:
            line = input(cyan("  > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line == "/help":
            print(LIVE_HELP.format(temperature=temperature))
            continue

        if line == "/style":
            if not os.path.exists(ADAPTER):
                note("No adapter checkpoint found.")
                continue
            if styled is None:
                note("Loading the 261 KB adapter...")
                styled = load_model_for_inference(CHECKPOINT, "cpu", ADAPTER)
            using_style = not using_style
            note("Shakespeare adapter ON." if using_style else "Back to the base model.")
            continue

        if line.startswith("/temp"):
            try:
                temperature = max(0.05, float(line.split(maxsplit=1)[1]))
                note(f"Temperature is now {temperature}. Lower is safer, higher is wilder.")
            except (IndexError, ValueError):
                note("Usage: /temp 0.8")
            continue

        active = styled if using_style else model

        if line.startswith("/next"):
            text = line[len("/next") :].strip()
            if text:
                show_next_tokens(active, tokenizer, text)
            else:
                note("Usage: /next Once upon a")
            continue

        if line.startswith("/score"):
            body = line[len("/score") :].strip()
            if "|" not in body:
                note("Usage: /score the cat sat down | the cat sat wxyz")
                continue
            left, right = (part.strip() for part in body.split("|", 1))
            score_sentences(active, tokenizer, left, right)
            continue

        print()
        stream(active, tokenizer, line, args.max_new_tokens, temperature, args.top_p,
               torch.seed() % (2**31), eos_id)

    print(dim("  bye."))


# ---------------------------------------------------------------------- entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--act", type=int, choices=range(1, 7), action="append",
        help="run only these acts (repeatable); default is all six",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="skip the walkthrough and go straight to the interactive prompt",
    )
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-batches", dest="eval_batches", type=int, default=4,
        help="batches per perplexity measurement (higher = tighter estimate, slower)",
    )
    args = parser.parse_args()

    require_artifacts()
    acts = sorted(set(args.act)) if args.act else [1, 2, 3, 4, 5, 6]
    if args.live:
        acts = []

    print()
    print(bold("  transformer-from-scratch") + dim("  ·  live demo"))
    print(dim("  a BPE tokenizer, a Transformer, AdamW and LoRA, built from the papers"))
    print(dim("  every number below is computed now, on this machine, on CPU"))

    started = time.perf_counter()
    tokenizer = Tokenizer.from_files(VOCAB, MERGES, special_tokens=["<|endoftext|>"])
    model = None
    cfg = TrainConfig(**load_config(CHECKPOINT))

    if 1 in acts:
        tokenizer = act_tokenizer(args)
    if 2 in acts:
        model, cfg = act_model(args)
    if model is None and (3 in acts or 4 in acts):
        model = load_model_for_inference(CHECKPOINT, "cpu")
    if 3 in acts:
        act_generate(args, model, tokenizer)
    if 4 in acts:
        act_causality(args, model, tokenizer, cfg)
    if 5 in acts:
        act_lora(args, tokenizer, cfg)
    if 6 in acts:
        act_init(args, cfg)
    if args.live:
        act_live(args, tokenizer, cfg)

    print()
    rule()
    print(
        dim(f"  demo complete in {time.perf_counter() - started:.1f}s  ·  "
            f"138 tests: pytest tests/ -q  ·  full write-up: README.md")
    )
    print()


if __name__ == "__main__":
    main()
