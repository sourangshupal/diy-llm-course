"""Week 2 Lab 2: Train a BPE tokenizer with Hugging Face `tokenizers`.

This script trains a small BPE tokenizer on a synthetic corpus and saves it
to disk so it can be reused in later weeks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def build_corpus(output_path: Path, num_sentences: int = 5_000) -> Path:
    """Create a small synthetic English corpus for tokenizer training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    words = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "cat", "dog", "house", "car", "tree", "river", "mountain", "city",
        "run", "jump", "walk", "swim", "fly", "think", "learn", "build",
        "quickly", "slowly", "carefully", "happily", "sadly", "beautifully",
        "large", "small", "fast", "slow", "bright", "dark", "happy", "sad",
    ]

    rng = __import__("random").Random(42)
    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(num_sentences):
            sentence = " ".join(rng.choices(words, k=rng.randint(5, 15)))
            f.write(sentence + "\n")
    return output_path


def train_tokenizer(
    corpus_path: Path,
    output_dir: Path,
    vocab_size: int = 2_000,
) -> Tokenizer:
    """Train a BPE tokenizer and save it to `output_dir`."""
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|pad|>", "<|endoftext|>", "<|unk|>"],
        min_frequency=2,
        show_progress=True,
    )

    print(f"Training tokenizer on {corpus_path}...")
    tokenizer.train([str(corpus_path)], trainer)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")

    # Save a small metadata file for human inspection
    metadata = {
        "vocab_size": vocab_size,
        "corpus_path": str(corpus_path),
        "num_special_tokens": len(["<|pad|>", "<|endoftext|>", "<|unk|>"]),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return tokenizer


def main() -> int:
    """Train a tokenizer and run a quick encode/decode sanity check."""
    corpus_path = Path("week02/data/tiny_corpus.txt")
    output_dir = Path("week02/data/my_tokenizer")

    corpus_path = build_corpus(corpus_path)
    tokenizer = train_tokenizer(corpus_path, output_dir, vocab_size=2_000)

    text = "the happy dog runs quickly"
    encoded = tokenizer.encode(text)
    print(f"\nInput:  {text!r}")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs:    {encoded.ids}")
    print(f"Decoded: {tokenizer.decode(encoded.ids)!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
