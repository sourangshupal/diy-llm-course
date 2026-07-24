"""Week 4 Lab: text generation from a trained mini Transformer.

Supports greedy decoding, temperature scaling, top-k, and nucleus (top-p) sampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data import CharDataset, build_corpus_file
from model import TransformerConfig, TransformerLM


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[TransformerLM, CharDataset]:
    """Load a trained model and build a matching tokenizer from the training corpus."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = TransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Rebuild tokenizer from training data (naive but sufficient for the char-level demo)
    config_dict = vars(config)
    data_path = Path("week04/data/corpus.txt")
    if not data_path.exists():
        build_corpus_file(data_path, repeats=100)
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    dataset = CharDataset(text, seq_len=config_dict.get("max_seq_len", 512))
    return model, dataset


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Sample one token from logits with optional temperature, top-k, and top-p."""
    logits = logits / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1)

    if top_k is not None and top_k > 0:
        top_k = min(top_k, probs.size(-1))
        indices_to_remove = probs < torch.topk(probs, top_k).values[..., -1, None]
        probs = probs.masked_fill(indices_to_remove, 0.0)
        probs = probs / probs.sum()

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        probs = probs.masked_fill(indices_to_remove, 0.0)
        probs = probs / probs.sum()

    return torch.multinomial(probs, num_samples=1).item()


def generate(
    model: TransformerLM,
    tokenizer: CharDataset,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generate text continuation from a prompt."""
    input_ids = tokenizer.tokenizer.encode(prompt)
    if not input_ids:
        return ""

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop to max_seq_len
        if input_tensor.size(1) > model.config.max_seq_len:
            input_tensor = input_tensor[:, -model.config.max_seq_len :]

        with torch.no_grad():
            logits = model(input_tensor)
        next_logits = logits[0, -1, :]
        next_token = sample_next_token(next_logits, temperature, top_k, top_p)
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)

    output_ids = input_tensor[0].tolist()
    return tokenizer.tokenizer.decode(output_ids)


def main() -> int:
    """Generate text from a checkpoint."""
    parser = argparse.ArgumentParser(description="Generate text from a trained mini Transformer")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--prompt", type=str, default="The quick ", help="Generation prompt")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, device)

    print(f"Prompt: {args.prompt!r}")
    output = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
