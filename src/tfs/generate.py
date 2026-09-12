"""Autoregressive sampling from a trained checkpoint, with optional LoRA adapter."""

from __future__ import annotations

import argparse

import torch

from .checkpoint import load_checkpoint, load_config
from .nn import inject_lora, softmax
from .tokenizer.bpe import Tokenizer
from .train import TrainConfig, build_model


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose mass reaches top_p."""
    if not 0.0 < top_p < 1.0:
        return probs

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Subtracting the token's own mass keeps the first token even if it alone exceeds top_p.
    keep = (cumulative - sorted_probs) < top_p

    filtered = torch.zeros_like(probs)
    filtered[sorted_idx[keep]] = sorted_probs[keep]
    return filtered / filtered.sum()


@torch.no_grad()
def generate(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: str = "cpu",
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)

    for _ in range(max_new_tokens):
        context = ids[-model.max_seq_len :]
        logits = model(torch.tensor([context], dtype=torch.long, device=device))[0, -1]

        if temperature <= 0.0:
            next_id = int(logits.argmax())
        else:
            probs = top_p_filter(softmax(logits / temperature, dim=-1), top_p)
            next_id = int(torch.multinomial(probs, num_samples=1, generator=generator))

        if eos_id is not None and next_id == eos_id:
            break
        ids.append(next_id)

    return tokenizer.decode(ids)


def load_model_for_inference(checkpoint: str, device: str, adapter: str | None = None):
    saved = load_config(checkpoint, device=device)
    if saved is None:
        raise ValueError(f"{checkpoint} has no embedded config; cannot rebuild the model")
    model = build_model(TrainConfig(**saved), device)

    if adapter is None:
        load_checkpoint(checkpoint, model, device=device)
        return model

    payload = torch.load(adapter, map_location=device, weights_only=False)
    load_checkpoint(checkpoint, model, device=device)
    inject_lora(model, payload["rank"], payload["alpha"], tuple(payload["target_modules"]))
    missing, unexpected = model.load_state_dict(payload["lora"], strict=False)
    if unexpected:
        raise ValueError(f"adapter has tensors that don't fit this model: {unexpected}")
    model.to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True, help="vocab.pkl from data preparation")
    parser.add_argument("--merges", required=True, help="merges.pkl from data preparation")
    parser.add_argument("--adapter", help="optional LoRA adapter.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=["<|endoftext|>"])
    model = load_model_for_inference(args.checkpoint, device, args.adapter)
    eos_id = tokenizer.byte_to_id.get("<|endoftext|>".encode("utf-8"))

    generator = torch.Generator(device=device).manual_seed(args.seed)
    for i in range(args.num_samples):
        text = generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            eos_id=eos_id,
            generator=generator,
        )
        if args.num_samples > 1:
            print(f"\n--- sample {i + 1} ---")
        print(text)


if __name__ == "__main__":
    main()
