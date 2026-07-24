"""Week 2 Lab 3: Inspect a trained BPE tokenizer.

Loads a saved tokenizer and lets students explore:
  - vocabulary size and special tokens,
  - learned merge rules,
  - tokenization length for different inputs,
  - tokenization bias across languages.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    """Load a tokenizer saved by `train_hf_tokenizer.py`."""
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"No tokenizer found at {tokenizer_path}. Run train_hf_tokenizer.py first."
        )
    return Tokenizer.from_file(str(tokenizer_path))


def analyze(tokenizer: Tokenizer) -> None:
    """Print basic statistics about a tokenizer."""
    vocab = tokenizer.get_vocab()
    print("=" * 60)
    print("Tokenizer Analysis")
    print("=" * 60)
    print(f"Vocabulary size: {len(vocab)}")

    # Special tokens
    special_tokens = [tok for tok in vocab if tok.startswith("<") and tok.endswith(">")]
    print(f"Special tokens: {special_tokens}")

    # Longest and shortest tokens
    sorted_by_len = sorted(vocab.items(), key=lambda kv: (-len(kv[0]), kv[0]))
    print(f"\nLongest tokens:")
    for tok, _id in sorted_by_len[:5]:
        print(f"  {tok!r} (len={len(tok)})")

    print(f"\nShortest tokens:")
    for tok, _id in sorted_by_len[-5:]:
        print(f"  {tok!r}")


def compare_texts(tokenizer: Tokenizer, texts: list[str]) -> None:
    """Encode several texts and compare token counts."""
    print("\n" + "=" * 60)
    print("Encode/Decode Comparison")
    print("=" * 60)
    print(f"{'Text':<40} {'#tokens':>8} {'#chars':>8}")
    print("-" * 60)
    for text in texts:
        encoded = tokenizer.encode(text)
        num_tokens = len(encoded.ids)
        num_chars = len(text)
        print(f"{text[:38]:<40} {num_tokens:>8} {num_chars:>8}")


def main() -> int:
    """Inspect the tokenizer saved in the data directory."""
    tokenizer_dir = Path("week02/data/my_tokenizer")
    tokenizer = load_tokenizer(tokenizer_dir)
    analyze(tokenizer)

    sample_texts = [
        "the quick brown fox jumps over the lazy dog",
        "Hello, world!",
        "Mathematics: 1 + 2 = 3",
        "你好世界",  # Chinese
        "🐋 whales are large marine mammals",
    ]
    compare_texts(tokenizer, sample_texts)

    # Demonstrate decoding
    example = sample_texts[0]
    encoded = tokenizer.encode(example)
    print("\nDetailed example:")
    print(f"  Input:   {example!r}")
    print(f"  Tokens:  {encoded.tokens}")
    print(f"  IDs:     {encoded.ids}")
    print(f"  Decode:  {tokenizer.decode(encoded.ids)!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
