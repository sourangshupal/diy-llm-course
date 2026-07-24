"""Week 2 Lab 1: Byte-Pair Encoding from scratch.

This is a minimal, pure-Python implementation of BPE intended for teaching.
It is intentionally slow and simple so every step of the algorithm is visible.
"""

from __future__ import annotations

import collections
from typing import Iterable


def get_stats(vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
    """Count frequencies of adjacent symbol pairs in the vocabulary."""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = list(word)
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return dict(pairs)


def merge_vocab(pair: tuple[str, str], vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
    """Merge all occurrences of `pair` in every word of the vocabulary."""
    bigram = " ".join(pair)
    replacement = "".join(pair)
    new_vocab: dict[tuple[str, ...], int] = {}
    for word, freq in vocab.items():
        symbols = list(word)
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            # Check if the current and next symbol form the target pair
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                new_symbols.append(replacement)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab


def train_bpe(
    corpus: Iterable[str],
    num_merges: int = 10,
    end_of_word: str = "</w>",
) -> tuple[dict[tuple[str, ...], int], list[tuple[str, str]]]:
    """Train BPE on a whitespace-tokenized corpus.

    Args:
        corpus: list of raw text strings.
        num_merges: number of merge operations to perform.
        end_of_word: special symbol appended to each word to mark boundaries.

    Returns:
        (final vocabulary mapping token tuples to frequencies,
         list of merges performed).
    """
    # Build initial word-frequency vocabulary
    word_freqs: collections.Counter = collections.Counter()
    for text in corpus:
        for word in text.strip().split():
            word_freqs[word] += 1

    # Convert to symbol tuples with end-of-word marker
    vocab: dict[tuple[str, ...], int] = {}
    for word, freq in word_freqs.items():
        symbols = tuple(list(word) + [end_of_word])
        vocab[symbols] = freq

    merges: list[tuple[str, str]] = []
    for step in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)  # type: ignore[arg-type]
        vocab = merge_vocab(best, vocab)
        merges.append(best)
        print(f"Merge {step + 1}: {best!r} -> {''.join(best)!r} (freq {pairs[best]})")

    return vocab, merges


def encode_word(word: str, merges: list[tuple[str, str]], end_of_word: str = "</w>") -> list[str]:
    """Encode a single word using a learned list of BPE merges."""
    symbols = list(word) + [end_of_word]
    for first, second in merges:
        replacement = first + second
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                new_symbols.append(replacement)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols


def encode(text: str, merges: list[tuple[str, str]], end_of_word: str = "</w>") -> list[str]:
    """Encode a full text string using learned BPE merges."""
    tokens: list[str] = []
    for word in text.strip().split():
        tokens.extend(encode_word(word, merges, end_of_word))
    return tokens


def main() -> int:
    """Run a tiny BPE training demo."""
    corpus = [
        "low lower lowest",
        "new newer newest",
        "wide wider widest",
    ]

    print("Training BPE from scratch on a tiny corpus...")
    print("=" * 60)
    vocab, merges = train_bpe(corpus, num_merges=10)

    print("\nFinal vocabulary (sample):")
    for tokens, freq in sorted(vocab.items(), key=lambda x: -x[1])[:10]:
        print(f"  {' '.join(tokens)!r}: {freq}")

    print("\nEncoding example:")
    example = "lowest"
    encoded = encode(example, merges)
    print(f"  '{example}' -> {encoded}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
